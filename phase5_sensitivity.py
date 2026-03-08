from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2_simulation import read_and_prepare_data
from phase5_common import (
    DEFAULT_CELL_CONFIG_JSON_PATH,
    DIFFUSION_CANDIDATES,
    OHMIC_CANDIDATES,
    REACTION_CANDIDATES,
    build_rmse_weights,
    downsample_for_fitting,
    filter_cell_cycles,
    parameter_name_from_selector,
    parse_cycle_range,
    prepare_base_parameter_context,
    print_phase5_context,
    resolve_capacity_and_initial_soc,
    simulate_voltage_trace,
    validate_base_inputs,
)


INPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
OUTPUT_DIR = Path("outputs/phase5")
SENSITIVITY_TAIL_DOWNSAMPLE_STRIDE = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 sensitivity-only runner: evaluate parameter sensitivity and save "
            "window-RMSE tables plus voltage overlay plots."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--cell-config-json",
        type=Path,
        default=DEFAULT_CELL_CONFIG_JSON_PATH,
        help="JSON mapping cell IDs to capacity_ah and initial_soc.",
    )
    parser.add_argument("--cell", required=True, help="Target cell ID (e.g., CELL_A).")
    parser.add_argument(
        "--cycle",
        type=str,
        default=None,
        help="Optional cycle index or range (e.g., 2 or 1-2).",
    )
    parser.add_argument("--capacity-ah", type=float, default=None)
    parser.add_argument("--initial-soc", type=float, default=None)
    parser.add_argument("--model-name", choices=["SPM", "SPMe", "DFN"], default="SPMe")
    parser.add_argument("--parameter-set", default="Chen2020")
    parser.add_argument("--solver-mode", choices=["safe", "fast"], default="safe")
    parser.add_argument("--voltage-max", type=float, default=4.5)
    parser.add_argument("--voltage-min", type=float, default=2.0)
    parser.add_argument(
        "--ohmic-parameter",
        choices=["n", "p", "e"],
        default=None,
        help="Ohmic selector when not scanning all candidates.",
    )
    parser.add_argument(
        "--kinetic-parameter",
        choices=["n", "p", "e"],
        default=None,
        help="Kinetic selector when not scanning all candidates (e is invalid).",
    )
    parser.add_argument(
        "--diffusion-parameter",
        choices=["n", "p", "e"],
        default=None,
        help="Diffusion selector when not scanning all candidates.",
    )
    parser.add_argument(
        "--rmse-weights",
        nargs=4,
        type=float,
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=("OHMIC", "KINETIC", "DIFFUSION", "CAPACITY"),
        help="Weights for windowed delta-RMSE sensitivity score.",
    )
    parser.add_argument(
        "--sensitivity-scales",
        nargs="+",
        type=float,
        default=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        help="Multipliers used in one-at-a-time sensitivity scans.",
    )
    parser.add_argument(
        "--sensitivity-include-all-candidates",
        action="store_true",
        help="Scan all fixed candidates in each category.",
    )
    return parser.parse_args()


def _safe_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))


def save_sensitivity_voltage_plot(
    baseline_trace: pd.DataFrame,
    variant_traces: list[tuple[str, pd.DataFrame]],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    t_h_base = baseline_trace["time_s"].to_numpy(dtype=float) / 3600.0
    v_base = baseline_trace["voltage_sim_V"].to_numpy(dtype=float)
    ax.plot(t_h_base, v_base, label="baseline", linewidth=1.5, color="black")
    for label, trace in variant_traces:
        t_h = trace["time_s"].to_numpy(dtype=float) / 3600.0
        v = trace["voltage_sim_V"].to_numpy(dtype=float)
        ax.plot(t_h, v, label=label, linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Simulated voltage [V]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _window_delta_metrics(
    baseline_trace: pd.DataFrame,
    variant_trace: pd.DataFrame,
) -> dict[str, float]:
    dv = (
        variant_trace["voltage_sim_V"].to_numpy(dtype=float)
        - baseline_trace["voltage_sim_V"].to_numpy(dtype=float)
    )
    t = baseline_trace["time_in_step_s"].to_numpy(dtype=float)
    # Keep window definitions unchanged, but downsample only 120s+ region.
    idx_df = pd.DataFrame(
        {
            "time_in_step_s": t,
            "_row_id": np.arange(len(t), dtype=int),
        }
    )
    idx_downsampled = downsample_for_fitting(
        idx_df,
        tail_stride=SENSITIVITY_TAIL_DOWNSAMPLE_STRIDE,
    )
    keep_tail_mask = np.zeros(len(t), dtype=bool)
    keep_tail_mask[idx_downsampled["_row_id"].to_numpy(dtype=int)] = True

    def delta_rmse(mask: np.ndarray) -> float:
        vals = dv[mask]
        if vals.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(vals))))

    return {
        "delta_rmse_ohmic_0_2s_V": delta_rmse((t >= 0.0) & (t < 2.0)),
        "delta_rmse_kinetic_2_20s_V": delta_rmse((t >= 2.0) & (t < 20.0)),
        "delta_rmse_diffusion_20_120s_V": delta_rmse((t >= 20.0) & (t < 120.0)),
        "delta_rmse_capacity_120plus_s_V": delta_rmse((t >= 120.0) & keep_tail_mask),
        "delta_rmse_full_profile_V": float(np.sqrt(np.mean(np.square(dv)))),
    }


def _weighted_delta_score(
    metrics: dict[str, float],
    rmse_weights: dict[str, float],
) -> float:
    pairs = [
        ("delta_rmse_ohmic_0_2s_V", rmse_weights["ohmic"]),
        ("delta_rmse_kinetic_2_20s_V", rmse_weights["kinetic"]),
        ("delta_rmse_diffusion_20_120s_V", rmse_weights["diffusion"]),
        ("delta_rmse_capacity_120plus_s_V", rmse_weights["capacity"]),
    ]
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, w in pairs:
        value = float(metrics[name])
        if not np.isfinite(value):
            continue
        weighted_sum += float(w) * value
        weight_sum += float(w)
    if weight_sum <= 0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def resolve_candidate_list(
    category: str,
    args: argparse.Namespace,
) -> list[str]:
    if category == "ohmic":
        if args.sensitivity_include_all_candidates:
            return list(OHMIC_CANDIDATES)
        if args.ohmic_parameter is None:
            return []
        return [parameter_name_from_selector("ohmic", str(args.ohmic_parameter))]
    if category == "kinetic":
        if args.sensitivity_include_all_candidates:
            return list(REACTION_CANDIDATES)
        if args.kinetic_parameter is None:
            return []
        return [parameter_name_from_selector("kinetic", str(args.kinetic_parameter))]
    if category == "diffusion":
        if args.sensitivity_include_all_candidates:
            return list(DIFFUSION_CANDIDATES)
        if args.diffusion_parameter is None:
            return []
        return [parameter_name_from_selector("diffusion", str(args.diffusion_parameter))]
    raise ValueError(f"Unknown category: {category}")


def run_sensitivity_analysis(
    args: argparse.Namespace,
    cell_df: pd.DataFrame,
    base_value_map: dict[str, object],
    rmse_weights: dict[str, float],
    output_dir: Path,
    cell_id: str,
    cycle_range: tuple[int, int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    baseline_trace = simulate_voltage_trace(
        cell_df=cell_df,
        model_name=args.model_name,
        parameter_set=args.parameter_set,
        solver_mode=args.solver_mode,
        voltage_max=args.voltage_max,
        voltage_min=args.voltage_min,
        capacity_ah=float(args.capacity_ah),
        initial_soc=float(args.initial_soc),
        parameter_overrides=None,
        parameter_scales={},
        base_value_map=base_value_map,
    )

    plot_dir = output_dir / f"{_safe_name(cell_id)}_sensitivity_plots"
    cycle_label = (
        "all_cycles"
        if cycle_range is None
        else (
            f"cycle_{cycle_range[0]}"
            if cycle_range[0] == cycle_range[1]
            else f"cycles_{cycle_range[0]}-{cycle_range[1]}"
        )
    )

    def append_result(
        category: str,
        parameter_name: str,
        test_value: float,
        test_label: str,
        metrics: dict[str, float],
        simulation_status: str = "ok",
        simulation_error: str = "",
    ) -> None:
        rows.append(
            {
                "category": category,
                "parameter_name": parameter_name,
                "test_value": test_value,
                "test_label": test_label,
                "simulation_status": simulation_status,
                "simulation_error": simulation_error,
                **metrics,
                "delta_weighted_score_V": _weighted_delta_score(metrics, rmse_weights),
            }
        )

    def append_failed_result(
        category: str,
        parameter_name: str,
        test_value: float,
        test_label: str,
        exc: Exception,
    ) -> None:
        print(
            f"Warning: skipped failed sensitivity variant "
            f"[{category}] {parameter_name} ({test_label}): {exc}"
        )
        append_result(
            category=category,
            parameter_name=parameter_name,
            test_value=test_value,
            test_label=test_label,
            metrics={
                "delta_rmse_ohmic_0_2s_V": float("nan"),
                "delta_rmse_kinetic_2_20s_V": float("nan"),
                "delta_rmse_diffusion_20_120s_V": float("nan"),
                "delta_rmse_capacity_120plus_s_V": float("nan"),
                "delta_rmse_full_profile_V": float("nan"),
            },
            simulation_status="failed",
            simulation_error=str(exc),
        )

    baseline_metrics = {
        "delta_rmse_ohmic_0_2s_V": 0.0,
        "delta_rmse_kinetic_2_20s_V": 0.0,
        "delta_rmse_diffusion_20_120s_V": 0.0,
        "delta_rmse_capacity_120plus_s_V": 0.0,
        "delta_rmse_full_profile_V": 0.0,
    }
    append_result("baseline", "baseline", 1.0, "baseline", baseline_metrics)

    by_category_candidates = {
        "ohmic": resolve_candidate_list("ohmic", args),
        "kinetic": resolve_candidate_list("kinetic", args),
        "diffusion": resolve_candidate_list("diffusion", args),
    }
    for category, candidates in by_category_candidates.items():
        for parameter_name in candidates:
            if parameter_name not in base_value_map:
                continue
            variants: list[tuple[str, pd.DataFrame]] = []
            for scale in args.sensitivity_scales:
                try:
                    variant = simulate_voltage_trace(
                        cell_df=cell_df,
                        model_name=args.model_name,
                        parameter_set=args.parameter_set,
                        solver_mode=args.solver_mode,
                        voltage_max=args.voltage_max,
                        voltage_min=args.voltage_min,
                        capacity_ah=float(args.capacity_ah),
                        initial_soc=float(args.initial_soc),
                        parameter_overrides=None,
                        parameter_scales={parameter_name: float(scale)},
                        base_value_map=base_value_map,
                    )
                except Exception as exc:
                    append_failed_result(
                        category=category,
                        parameter_name=parameter_name,
                        test_value=float(scale),
                        test_label=f"scale_{scale}",
                        exc=exc,
                    )
                    continue
                metrics = _window_delta_metrics(baseline_trace, variant)
                if float(scale) != 1.0:
                    variants.append((f"{parameter_name} x{float(scale):.3g}", variant))
                append_result(category, parameter_name, float(scale), f"scale_{scale}", metrics)
            if variants:
                save_sensitivity_voltage_plot(
                    baseline_trace=baseline_trace,
                    variant_traces=variants,
                    output_path=plot_dir
                    / f"{cycle_label}_{_safe_name(category)}_{_safe_name(parameter_name)}.png",
                    title=f"{cell_id} sensitivity: {category} | {parameter_name}",
                )

    details_df = pd.DataFrame(rows)
    ranked_rows: list[dict[str, object]] = []
    grouped = details_df[details_df["category"] != "baseline"].groupby(
        ["category", "parameter_name"], dropna=False
    )
    for (category, parameter_name), grp in grouped:
        score_col = "delta_weighted_score_V"
        valid_scores = grp[np.isfinite(grp[score_col].to_numpy(dtype=float))]
        if valid_scores.empty:
            continue
        best_idx = valid_scores[score_col].idxmax()
        best_row = valid_scores.loc[best_idx]
        spread = float(valid_scores[score_col].max() - valid_scores[score_col].min())
        ranked_rows.append(
            {
                "category": category,
                "parameter_name": parameter_name,
                "best_test_label": best_row["test_label"],
                "best_test_value": float(best_row["test_value"]),
                "best_score_column": score_col,
                "best_score_V": float(best_row[score_col]),
                "score_spread_V": spread,
            }
        )
    ranked_df = pd.DataFrame(ranked_rows).sort_values(
        ["category", "best_score_V", "score_spread_V"], ascending=[True, False, False]
    )
    return details_df, ranked_df
def main() -> None:
    args = parse_args()
    cycle_range = parse_cycle_range(args.cycle)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capacity_ah, initial_soc = resolve_capacity_and_initial_soc(
        cell_config_json=args.cell_config_json,
        cell_id=args.cell,
        capacity_ah=args.capacity_ah,
        initial_soc=args.initial_soc,
    )
    args.capacity_ah = float(capacity_ah)
    args.initial_soc = float(initial_soc)
    validate_base_inputs(capacity_ah=float(args.capacity_ah), initial_soc=float(args.initial_soc))
    if len(args.sensitivity_scales) == 0:
        raise ValueError("--sensitivity-scales must include at least one value.")
    rmse_weights = build_rmse_weights(list(args.rmse_weights))

    df = read_and_prepare_data(args.input_csv)
    cell_df = filter_cell_cycles(df=df, cell_id=args.cell, cycle_range=cycle_range)

    prepared_names = set(DIFFUSION_CANDIDATES + REACTION_CANDIDATES + OHMIC_CANDIDATES)
    base_value_map, missing_names = prepare_base_parameter_context(
        parameter_set=args.parameter_set,
        candidate_names=prepared_names,
    )
    print_phase5_context(
        target_label="Phase5 sensitivity target",
        cell_id=args.cell,
        cycle_range=cycle_range,
        model_name=args.model_name,
        parameter_set=args.parameter_set,
        n_points=len(cell_df),
        rmse_weights=rmse_weights,
        base_value_map=base_value_map,
        missing_names=missing_names,
    )

    details_df, ranked_df = run_sensitivity_analysis(
        args=args,
        cell_df=cell_df,
        base_value_map=base_value_map,
        rmse_weights=rmse_weights,
        output_dir=args.output_dir,
        cell_id=args.cell,
        cycle_range=cycle_range,
    )
    details_path = args.output_dir / f"{args.cell}_sensitivity_details.csv"
    ranked_path = args.output_dir / f"{args.cell}_sensitivity_ranked.csv"
    details_df.to_csv(details_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)
    print(f"Saved sensitivity details: {details_path}")
    print(f"Saved sensitivity ranking: {ranked_path}")


if __name__ == "__main__":
    main()
