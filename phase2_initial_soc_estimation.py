from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm


INPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
OUTPUT_DIR = Path("outputs/phase2")
DEFAULT_CAPACITY_CSV = OUTPUT_DIR / "capacity_estimates.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-cell initial SoC by fitting first-step voltage with PyBaMM SPMe "
            "using bisection over initial SoC."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument(
        "--cell-config-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping cell IDs to capacity_ah, e.g. "
            '{"CELL_A":{"capacity_ah":5.1}}. If provided, this is used for capacities.'
        ),
    )
    parser.add_argument(
        "--capacity-estimates-csv",
        type=Path,
        default=DEFAULT_CAPACITY_CSV,
        help="CSV with columns: cell_id, estimated_capacity_ah.",
    )
    parser.add_argument(
        "--parameter-set",
        choices=["Chen2020"],
        default="Chen2020",
    )
    parser.add_argument("--solver-mode", choices=["safe", "fast"], default="safe")
    parser.add_argument("--soc-lower", type=float, default=0.01)
    parser.add_argument("--soc-upper", type=float, default=0.99)
    parser.add_argument(
        "--soc-tol",
        type=float,
        default=1e-4,
        help="Stop when bracket width is below this value.",
    )
    parser.add_argument(
        "--voltage-tol",
        type=float,
        default=1e-3,
        help="Stop when absolute end-voltage error [V] is below this value.",
    )
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument(
        "--fallback-grid-size",
        type=int,
        default=25,
        help="Grid size for fallback search when bisection cannot be bracketed.",
    )
    return parser.parse_args()


def read_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    required = {"datetime", "cell_id", "cycle", "step", "current_A", "voltage_V"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "cell_id", "cycle", "step", "current_A", "voltage_V"]).copy()
    df = df.sort_values(["cell_id", "datetime"]).reset_index(drop=True)
    return df


def load_capacities(
    cell_config_json: Path | None,
    capacity_estimates_csv: Path,
) -> dict[str, float]:
    if cell_config_json is not None:
        if not cell_config_json.exists():
            raise FileNotFoundError(f"cell-config-json not found: {cell_config_json}")
        with cell_config_json.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        cap_map: dict[str, float] = {}
        for cell_id, cfg in raw.items():
            if not isinstance(cfg, dict) or "capacity_ah" not in cfg:
                raise ValueError(f"{cell_id} config must include capacity_ah.")
            cap_map[str(cell_id)] = float(cfg["capacity_ah"])
        return cap_map

    if not capacity_estimates_csv.exists():
        raise FileNotFoundError(
            f"capacity-estimates-csv not found: {capacity_estimates_csv}. "
            "Run phase2_capacity_estimation.py first or pass --cell-config-json."
        )
    cap_df = pd.read_csv(capacity_estimates_csv)
    required = {"cell_id", "estimated_capacity_ah"}
    missing = sorted(required - set(cap_df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {capacity_estimates_csv}: {missing}")
    return {
        str(row["cell_id"]): float(row["estimated_capacity_ah"])
        for _, row in cap_df.iterrows()
    }


def first_step_segment(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    ordered = cell_df.sort_values("datetime").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("Cell dataframe is empty.")

    first_cycle = float(ordered.iloc[0]["cycle"])
    first_step = float(ordered.iloc[0]["step"])

    mask = (
        np.isclose(ordered["cycle"].to_numpy(dtype=float), first_cycle, atol=1e-9)
        & np.isclose(ordered["step"].to_numpy(dtype=float), first_step, atol=1e-9)
    )

    end_idx = 0
    while end_idx < len(mask) and bool(mask[end_idx]):
        end_idx += 1

    seg = ordered.iloc[:end_idx].copy()
    if len(seg) < 2:
        raise ValueError("First step segment has fewer than 2 points.")
    return seg, first_cycle, first_step


def make_model() -> pybamm.BaseModel:
    return pybamm.lithium_ion.SPMe()


def run_first_step_voltage(
    seg: pd.DataFrame,
    capacity_ah: float,
    initial_soc: float,
    parameter_set: str,
    solver_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_s = (seg["datetime"] - seg["datetime"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    i_exp = seg["current_A"].to_numpy(dtype=float)
    v_exp = seg["voltage_V"].to_numpy(dtype=float)

    model = make_model()
    params = pybamm.ParameterValues(parameter_set)
    params.update({"Nominal cell capacity [A.h]": float(capacity_ah)})

    params.update({"Current function [A]": pybamm.Interpolant(t_s, i_exp, pybamm.t)})
    sim = pybamm.Simulation(
        model=model,
        parameter_values=params,
        solver=pybamm.CasadiSolver(mode=solver_mode),
    )
    sol = sim.solve(t_eval=t_s, initial_soc=float(initial_soc))
    v_sim = np.asarray(sol["Terminal voltage [V]"].entries, dtype=float)
    return t_s, v_exp, v_sim


def objective_end_voltage_error(
    seg: pd.DataFrame,
    capacity_ah: float,
    soc: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    _, v_exp, v_sim = run_first_step_voltage(
        seg=seg,
        capacity_ah=capacity_ah,
        initial_soc=soc,
        parameter_set=args.parameter_set,
        solver_mode=args.solver_mode,
    )
    end_err = float(v_sim[-1] - v_exp[-1])
    rmse = float(np.sqrt(np.mean((v_sim - v_exp) ** 2)))
    return end_err, rmse


def safe_objective_eval(
    seg: pd.DataFrame,
    capacity_ah: float,
    soc: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    try:
        err, rmse = objective_end_voltage_error(seg, capacity_ah, soc, args)
        return {
            "valid": True,
            "soc": float(soc),
            "err": float(err),
            "rmse": float(rmse),
            "reason": "ok",
        }
    except pybamm.SolverError as exc:
        msg = str(exc)
        if "Maximum voltage [V]" in msg:
            reason = "max_voltage_violation"
        elif "Minimum voltage [V]" in msg:
            reason = "min_voltage_violation"
        else:
            reason = "solver_error"
        return {
            "valid": False,
            "soc": float(soc),
            "err": np.nan,
            "rmse": np.nan,
            "reason": reason,
            "message": msg,
        }


def estimate_soc_for_cell(
    seg: pd.DataFrame,
    capacity_ah: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    lo = float(args.soc_lower)
    hi = float(args.soc_upper)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("Require 0 <= soc-lower < soc-upper <= 1.")

    grid = np.linspace(lo, hi, max(3, int(args.fallback_grid_size)))
    evals = [safe_objective_eval(seg, capacity_ah, float(soc), args) for soc in grid]
    valid_evals = [ev for ev in evals if bool(ev["valid"])]

    if not valid_evals:
        reasons = sorted({str(ev.get("reason", "unknown")) for ev in evals})
        raise ValueError(
            "No feasible SoC found in search range. "
            f"Try narrowing soc bounds. reasons={reasons}"
        )

    # Prefer bisection if a valid sign-changing bracket exists among sampled points.
    bracket = None
    for left, right in zip(valid_evals[:-1], valid_evals[1:]):
        err_l = float(left["err"])
        err_r = float(right["err"])
        if err_l == 0.0:
            return {
                "estimated_initial_soc": float(left["soc"]),
                "method": "grid_exact",
                "iterations": len(grid),
                "end_voltage_error_V": err_l,
                "rmse_V": float(left["rmse"]),
            }
        if err_r == 0.0:
            return {
                "estimated_initial_soc": float(right["soc"]),
                "method": "grid_exact",
                "iterations": len(grid),
                "end_voltage_error_V": err_r,
                "rmse_V": float(right["rmse"]),
            }
        if (err_l < 0 < err_r) or (err_r < 0 < err_l):
            bracket = (left, right)
            break

    if bracket is None:
        best = min(valid_evals, key=lambda ev: abs(float(ev["err"])))
        return {
            "estimated_initial_soc": float(best["soc"]),
            "method": "grid_fallback",
            "iterations": len(grid),
            "end_voltage_error_V": float(best["err"]),
            "rmse_V": float(best["rmse"]),
        }

    lo_soc = float(bracket[0]["soc"])
    hi_soc = float(bracket[1]["soc"])
    err_lo = float(bracket[0]["err"])
    err_hi = float(bracket[1]["err"])
    _ = err_hi  # kept for clarity in sign-tracking logic

    best_soc = lo_soc
    best_err = err_lo
    best_rmse = float(bracket[0]["rmse"])

    for it in range(1, int(args.max_iter) + 1):
        mid = 0.5 * (lo_soc + hi_soc)
        mid_ev = safe_objective_eval(seg, capacity_ah, mid, args)

        if not bool(mid_ev["valid"]):
            reason = str(mid_ev["reason"])
            if reason == "max_voltage_violation":
                hi_soc = mid
            elif reason == "min_voltage_violation":
                lo_soc = mid
            else:
                # Unknown solver issue: shrink conservatively toward center.
                hi_soc = mid
            if (hi_soc - lo_soc) <= float(args.soc_tol):
                break
            continue

        err_mid = float(mid_ev["err"])
        rmse_mid = float(mid_ev["rmse"])
        best_soc = mid
        best_err = err_mid
        best_rmse = rmse_mid

        if abs(err_mid) <= float(args.voltage_tol) or (hi_soc - lo_soc) <= float(args.soc_tol):
            return {
                "estimated_initial_soc": mid,
                "method": "bisection",
                "iterations": it + len(grid),
                "end_voltage_error_V": err_mid,
                "rmse_V": rmse_mid,
            }

        if (err_lo < 0 < err_mid) or (err_mid < 0 < err_lo):
            hi_soc = mid
        else:
            lo_soc = mid
            err_lo = err_mid

    return {
        "estimated_initial_soc": best_soc,
        "method": "bisection_max_iter",
        "iterations": int(args.max_iter) + len(grid),
        "end_voltage_error_V": best_err,
        "rmse_V": best_rmse,
    }


def main() -> None:
    args = parse_args()
    df = read_data(args.input_csv)
    capacity_map = load_capacities(args.cell_config_json, args.capacity_estimates_csv)

    if args.cells:
        cells = args.cells
    else:
        cells = sorted(capacity_map.keys())

    out_rows: list[dict[str, object]] = []
    config_out: dict[str, dict[str, float]] = {}

    for cell_id in cells:
        if cell_id not in capacity_map:
            raise ValueError(f"No capacity found for {cell_id}.")
        cell_df = df[df["cell_id"] == cell_id].copy()
        if cell_df.empty:
            raise ValueError(f"Cell {cell_id} not found in input data.")

        seg, first_cycle, first_step = first_step_segment(cell_df)
        capacity_ah = float(capacity_map[cell_id])

        result = estimate_soc_for_cell(seg, capacity_ah, args)
        est_soc = float(result["estimated_initial_soc"])

        out_rows.append(
            {
                "cell_id": cell_id,
                "first_cycle": first_cycle,
                "first_step": first_step,
                "first_step_points": len(seg),
                "first_step_duration_s": float(
                    (seg["datetime"].iloc[-1] - seg["datetime"].iloc[0]).total_seconds()
                ),
                "capacity_ah": capacity_ah,
                "parameter_set": args.parameter_set,
                "estimated_initial_soc": est_soc,
                "method": result["method"],
                "iterations": result["iterations"],
                "end_voltage_error_V": result["end_voltage_error_V"],
                "rmse_V": result["rmse_V"],
            }
        )
        config_out[cell_id] = {"capacity_ah": capacity_ah, "initial_soc": est_soc}
        print(
            f"{cell_id}: estimated_initial_soc={est_soc:.6f}, "
            f"method={result['method']}, end_err={result['end_voltage_error_V']:.6f} V"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "initial_soc_estimates.csv"
    pd.DataFrame(out_rows).to_csv(summary_path, index=False)

    config_path = args.output_dir / "phase2_cell_config_with_estimated_soc.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)
        f.write("\n")

    print(f"Saved: {summary_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
