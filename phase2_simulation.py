from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm


INPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
OUTPUT_DIR = Path("outputs/phase2")
DEFAULT_MODEL_NAME = "SPMe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run battery-model simulations against cleaned experimental data for one or more "
            "cells and compare voltage traces."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV_PATH,
        help="Path to cleaned data CSV from Phase 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where Phase 2 outputs are saved.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Cell IDs to simulate (e.g., CELL_A CELL_B).",
    )
    parser.add_argument(
        "--cell-config-json",
        type=Path,
        default=None,
        help=(
            "JSON mapping cell IDs to capacity_ah and initial_soc. "
            'Example: {"CELL_A":{"capacity_ah":4.8,"initial_soc":0.95}}'
        ),
    )
    parser.add_argument(
        "--capacity-ah",
        type=float,
        default=None,
        help="Assumed nominal capacity in Ah for single-cell mode.",
    )
    parser.add_argument(
        "--initial-soc",
        type=float,
        default=None,
        help="Initial SoC in [0, 1] for single-cell mode.",
    )
    parser.add_argument(
        "--max-cycle",
        type=float,
        default=None,
        help="Optional inclusive upper cycle index to simulate (e.g., 50).",
    )
    parser.add_argument(
        "--model-name",
        choices=["SPM", "SPMe", "DFN"],
        default=DEFAULT_MODEL_NAME,
        help="PyBaMM lithium-ion model type.",
    )
    parser.add_argument(
        "--parameter-set",
        default="Chen2020",
        help="PyBaMM parameter set name (e.g., Chen2020, Chen2020_composite).",
    )
    parser.add_argument(
        "--solver-mode",
        choices=["safe", "fast"],
        default="safe",
        help="PyBaMM CasadiSolver mode.",
    )
    parser.add_argument(
        "--voltage-max",
        type=float,
        default=4.5,
        help="Optional upper voltage cut-off [V] override.",
    )
    parser.add_argument(
        "--voltage-min",
        type=float,
        default=2.0,
        help="Optional lower voltage cut-off [V] override.",
    )
    return parser.parse_args()


def read_and_prepare_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_columns = {"datetime", "cell_id", "cycle", "step", "current_A", "voltage_V"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {input_csv}: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["datetime", "cycle", "step", "current_A", "voltage_V"]).copy()
    df = df.sort_values(["cell_id", "datetime"]).reset_index(drop=True)
    return df


def load_cell_config(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Cell config JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("cell-config-json must be an object mapping cell_id -> config.")

    out: dict[str, dict[str, float]] = {}
    for cell_id, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Config for {cell_id} must be an object.")
        if "capacity_ah" not in cfg or "initial_soc" not in cfg:
            raise ValueError(f"Config for {cell_id} must include capacity_ah and initial_soc.")
        out[str(cell_id)] = {
            "capacity_ah": float(cfg["capacity_ah"]),
            "initial_soc": float(cfg["initial_soc"]),
        }
    return out


def determine_cells(
    df: pd.DataFrame,
    requested_cells: list[str] | None,
    config_map: dict[str, dict[str, float]],
) -> list[str]:
    available = set(df["cell_id"].unique())
    if requested_cells:
        cells = requested_cells
    elif config_map:
        cells = list(config_map.keys())
    else:
        cells = sorted(available)

    unknown = [cell for cell in cells if cell not in available]
    if unknown:
        raise ValueError(f"Requested cell IDs not found in data: {unknown}")

    return cells


def get_cell_initial_conditions(
    cell_id: str,
    config_map: dict[str, dict[str, float]],
    fallback_capacity_ah: float | None,
    fallback_initial_soc: float | None,
) -> tuple[float, float]:
    if cell_id in config_map:
        capacity_ah = config_map[cell_id]["capacity_ah"]
        initial_soc = config_map[cell_id]["initial_soc"]
    else:
        if fallback_capacity_ah is None or fallback_initial_soc is None:
            raise ValueError(
                f"Missing inputs for {cell_id}. Provide --cell-config-json or both "
                "--capacity-ah and --initial-soc for this cell."
            )
        capacity_ah = fallback_capacity_ah
        initial_soc = fallback_initial_soc

    if not (0.0 <= initial_soc <= 1.0):
        raise ValueError(f"initial_soc for {cell_id} must be in [0, 1], got {initial_soc}.")
    if capacity_ah <= 0:
        raise ValueError(f"capacity_ah for {cell_id} must be > 0, got {capacity_ah}.")
    return capacity_ah, initial_soc


def apply_capacity_scaling(
    parameter_values: pybamm.ParameterValues,
    target_capacity_ah: float,
) -> float:
    base_nominal_capacity = float(parameter_values["Nominal cell capacity [A.h]"])
    if base_nominal_capacity <= 0:
        raise ValueError(f"Invalid base nominal capacity: {base_nominal_capacity}")

    scale = float(target_capacity_ah) / base_nominal_capacity
    updates: dict[str, float] = {"Nominal cell capacity [A.h]": float(target_capacity_ah)}

    # For current-driven simulations, scaling parallel electrode count changes the
    # effective C-rate and makes the provided capacity materially affect dynamics.
    parallel_key = "Number of electrodes connected in parallel to make a cell"
    if parallel_key in parameter_values.keys():
        base_parallel = float(parameter_values[parallel_key])
        updates[parallel_key] = base_parallel * scale

    parameter_values.update(updates)
    return scale


def build_simulation(
    model: pybamm.BaseModel,
    parameter_values: pybamm.ParameterValues,
    time_s: np.ndarray,
    current_a: np.ndarray,
    solver_mode: str,
) -> pybamm.Simulation:
    current_input = pybamm.Interpolant(time_s, current_a, pybamm.t)
    parameter_values.update({"Current function [A]": current_input})
    solver = pybamm.CasadiSolver(mode=solver_mode)
    return pybamm.Simulation(model=model, parameter_values=parameter_values, solver=solver)


def make_model(model_name: str, parameter_set: str | None = None) -> pybamm.BaseModel:
    model_builders: dict[str, object] = {
        "SPM": pybamm.lithium_ion.SPM,
        "SPMe": pybamm.lithium_ion.SPMe,
        "DFN": pybamm.lithium_ion.DFN,
    }
    if model_name not in model_builders:
        raise ValueError(f"Unsupported model_name: {model_name}")

    builder = model_builders[model_name]
    # Composite parameter sets (e.g., Chen2020_composite) require multi-phase
    # negative electrode options in the model definition.
    if parameter_set is not None and "composite" in parameter_set.lower():
        return builder(options={"particle phases": ("2", "1")})
    return builder()


def validate_parameter_set_name(parameter_set: str) -> None:
    try:
        pybamm.ParameterValues(parameter_set)
    except Exception as exc:
        raise ValueError(
            f"Unknown or unsupported PyBaMM parameter set: {parameter_set}"
        ) from exc


def run_cell_simulation(
    cell_df: pd.DataFrame,
    model_name: str,
    parameter_set: str,
    capacity_ah: float,
    initial_soc: float,
    solver_mode: str,
    voltage_max: float | None,
    voltage_min: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, bool]:
    t_s = (cell_df["datetime"] - cell_df["datetime"].iloc[0]).dt.total_seconds().to_numpy()
    i_exp = cell_df["current_A"].to_numpy(dtype=float)
    v_exp = cell_df["voltage_V"].to_numpy(dtype=float)

    model = make_model(model_name, parameter_set=parameter_set)
    parameter_values = pybamm.ParameterValues(parameter_set)
    capacity_scale = apply_capacity_scaling(parameter_values, float(capacity_ah))
    if voltage_max is not None:
        parameter_values.update({"Upper voltage cut-off [V]": float(voltage_max)})
    if voltage_min is not None:
        parameter_values.update({"Lower voltage cut-off [V]": float(voltage_min)})

    sim = build_simulation(model, parameter_values, t_s, i_exp, solver_mode)
    initial_soc_applied = True
    try:
        solution = sim.solve(t_eval=t_s, initial_soc=float(initial_soc))
    except Exception as exc:
        # Composite parameter sets may not support the same eSOH-based
        # initial_soc path as non-composite sets. Retry without initial_soc.
        if "composite" not in parameter_set.lower():
            raise
        print(
            "Warning: initial_soc-based initialization failed for composite "
            f"parameter set ({parameter_set}); falling back to parameter defaults. "
            f"Original error: {exc}"
        )
        solution = sim.solve(t_eval=t_s)
        initial_soc_applied = False
    t_sim = np.asarray(solution["Time [s]"].entries, dtype=float)
    v_sim = solution["Terminal voltage [V]"].entries
    v_sim = np.asarray(v_sim, dtype=float)
    termination_reason = str(getattr(solution, "termination", "unknown"))
    return (
        t_s,
        i_exp,
        v_exp,
        t_sim,
        v_sim,
        capacity_scale,
        termination_reason,
        initial_soc_applied,
    )


def save_voltage_compare_plot(
    cell_id: str,
    model_name: str,
    t_exp: np.ndarray,
    v_exp: np.ndarray,
    t_sim: np.ndarray,
    v_sim: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_exp / 3600.0, v_exp, label="Experimental", linewidth=1.0)
    ax.plot(t_sim / 3600.0, v_sim, label=f"{model_name} simulated", linewidth=1.0)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Voltage [V]")
    title = f"{cell_id}: Experimental vs {model_name}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_early_termination_diagnostics(
    cell_id: str,
    model_name: str,
    t_exp: np.ndarray,
    v_exp: np.ndarray,
    t_sim: np.ndarray,
    v_sim: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_cell = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(cell_id))

    if len(t_sim) == 0:
        return

    # Always produce the standard comparison plot, even for early termination.
    save_voltage_compare_plot(
        cell_id=cell_id,
        model_name=model_name,
        t_exp=t_exp,
        v_exp=v_exp,
        t_sim=t_sim,
        v_sim=v_sim,
        output_path=output_dir / f"{safe_cell}_voltage_compare.png",
        title_suffix="early termination",
    )


def make_comparison_df(
    cell_id: str,
    cell_df: pd.DataFrame,
    t_s: np.ndarray,
    i_exp: np.ndarray,
    v_exp: np.ndarray,
    v_sim: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": cell_df["datetime"].to_numpy(),
            "cell_id": cell_id,
            "cycle": cell_df["cycle"].to_numpy() if "cycle" in cell_df.columns else np.nan,
            "step": cell_df["step"].to_numpy() if "step" in cell_df.columns else np.nan,
            "time_s": t_s,
            "current_A_sim_input": i_exp,
            "voltage_exp_V": v_exp,
            "voltage_sim_V": v_sim,
            "voltage_error_V": v_sim - v_exp,
        }
    )



def save_outputs(
    cell_id: str,
    model_name: str,
    comparison_df: pd.DataFrame,
    t_s: np.ndarray,
    v_exp: np.ndarray,
    v_sim: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_cell = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(cell_id))
    comparison_df.to_csv(output_dir / f"{safe_cell}_comparison.csv", index=False)

    save_voltage_compare_plot(
        cell_id=cell_id,
        model_name=model_name,
        t_exp=t_s,
        v_exp=v_exp,
        t_sim=t_s,
        v_sim=v_sim,
        output_path=output_dir / f"{safe_cell}_voltage_compare.png",
    )


def simulate_cells(
    df: pd.DataFrame,
    cells: list[str],
    config_map: dict[str, dict[str, float]],
    fallback_capacity_ah: float | None,
    fallback_initial_soc: float | None,
    output_dir: Path | None = None,
    max_cycle: float | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    parameter_set: str = "Chen2020",
    solver_mode: str = "safe",
    voltage_max: float | None = None,
    voltage_min: float | None = None,
    save_files: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if save_files and output_dir is None:
        raise ValueError("output_dir must be provided when save_files=True")

    summary_rows: list[dict[str, object]] = []
    comparison_by_cell: dict[str, pd.DataFrame] = {}

    for cell_id in cells:
        cell_df = df[df["cell_id"] == cell_id].copy()
        if max_cycle is not None:
            cell_df = cell_df[cell_df["cycle"] <= float(max_cycle)].copy()
        if cell_df.empty:
            raise ValueError(
                f"No data left for {cell_id} after applying max_cycle={max_cycle}."
            )
        capacity_ah, initial_soc = get_cell_initial_conditions(
            cell_id=cell_id,
            config_map=config_map,
            fallback_capacity_ah=fallback_capacity_ah,
            fallback_initial_soc=fallback_initial_soc,
        )

        print(
            f"Simulating {cell_id}: model={model_name}, parameter_set={parameter_set}, "
            f"capacity_ah={capacity_ah}, initial_soc={initial_soc}"
        )
        (
            t_s,
            i_exp,
            v_exp,
            t_sim,
            v_sim,
            capacity_scale,
            termination_reason,
            initial_soc_applied,
        ) = run_cell_simulation(
            cell_df=cell_df,
            model_name=model_name,
            parameter_set=parameter_set,
            capacity_ah=capacity_ah,
            initial_soc=initial_soc,
            solver_mode=solver_mode,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )
        print(f"{cell_id}: capacity scaling factor vs base set = {capacity_scale:.6f}")

        # Strict mode: simulation must cover full profile. If not, save diagnostics then fail.
        if len(v_sim) != len(v_exp) or len(t_sim) != len(t_s) or not np.isclose(
            t_sim[-1], t_s[-1], atol=1e-8, rtol=0.0
        ):
            if save_files and output_dir is not None:
                save_early_termination_diagnostics(
                    cell_id=cell_id,
                    model_name=model_name,
                    t_exp=t_s,
                    v_exp=v_exp,
                    t_sim=t_sim,
                    v_sim=v_sim,
                    output_dir=output_dir,
                )
            raise RuntimeError(
                "Simulation terminated before full current profile completed. "
                f"cell_id={cell_id}, sim_points={len(v_sim)}, exp_points={len(v_exp)}, "
                f"sim_end_s={t_sim[-1] if len(t_sim) else np.nan:.6f}, exp_end_s={t_s[-1]:.6f}. "
                f"termination_reason={termination_reason}. "
                f"Diagnostic files saved to {output_dir}."
            )
        comparison_df = make_comparison_df(
            cell_id=cell_id,
            cell_df=cell_df,
            t_s=t_s,
            i_exp=i_exp,
            v_exp=v_exp,
            v_sim=v_sim,
        )
        comparison_by_cell[cell_id] = comparison_df
        if save_files and output_dir is not None:
            save_outputs(
                cell_id=cell_id,
                model_name=model_name,
                comparison_df=comparison_df,
                t_s=t_s,
                v_exp=v_exp,
                v_sim=v_sim,
                output_dir=output_dir,
            )

        rmse = float(np.sqrt(np.mean((v_sim - v_exp) ** 2)))
        summary_rows.append(
            {
                "cell_id": cell_id,
                "model_name": model_name,
                "parameter_set": parameter_set,
                "capacity_ah": capacity_ah,
                "initial_soc": initial_soc,
                "initial_soc_applied": initial_soc_applied,
                "max_cycle_requested": max_cycle,
                "max_cycle_included": float(cell_df["cycle"].max()),
                "capacity_scale": capacity_scale,
                "solver_mode": solver_mode,
                "voltage_min": voltage_min,
                "voltage_max": voltage_max,
                "termination_reason": termination_reason,
                "rmse_V": rmse,
            }
        )
        print(f"Completed {cell_id}: RMSE={rmse:.6f} V")

    return pd.DataFrame(summary_rows), comparison_by_cell


def main() -> None:
    args = parse_args()
    validate_parameter_set_name(args.parameter_set)
    df = read_and_prepare_data(args.input_csv)
    config_map = load_cell_config(args.cell_config_json)
    cells = determine_cells(df, args.cells, config_map)

    if args.model_name == "DFN":
        max_points = int(df[df["cell_id"].isin(cells)].groupby("cell_id").size().max())
        if max_points > 50000:
            print(
                "Warning: DFN on very long profiles can be extremely slow "
                f"(up to {max_points} points per selected cell). "
                "If needed, try --solver-mode fast or use SPMe/SPM for screening."
            )

    try:
        summary_df, _ = simulate_cells(
            df=df,
            cells=cells,
            config_map=config_map,
            fallback_capacity_ah=args.capacity_ah,
            fallback_initial_soc=args.initial_soc,
            output_dir=args.output_dir,
        max_cycle=args.max_cycle,
            model_name=args.model_name,
            parameter_set=args.parameter_set,
            solver_mode=args.solver_mode,
            voltage_max=args.voltage_max,
            voltage_min=args.voltage_min,
            save_files=True,
        )
    except KeyboardInterrupt:
        print(
            "\nSimulation interrupted by user (KeyboardInterrupt). "
            "For DFN long runs, consider --solver-mode fast or SPMe/SPM."
        )
        raise
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_dir / "simulation_summary.csv", index=False)
    print(f"Saved summary: {args.output_dir / 'simulation_summary.csv'}")


if __name__ == "__main__":
    main()
