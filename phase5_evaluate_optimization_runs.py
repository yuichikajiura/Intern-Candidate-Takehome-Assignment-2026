from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DB_PATH = Path("outputs/phase4/battery_pipeline.db")
DEFAULT_OUTPUT_DIR = Path("outputs/phase5")

RMSE_COLUMNS = [
    "rmse_ohmic_0_2s_V",
    "rmse_kinetic_2_20s_V",
    "rmse_diffusion_20_120s_V",
    "rmse_capacity_120plus_s_V",
    "rmse_full_profile_V",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and compare Phase 5 optimization runs from DB. "
            "Outputs final parameter values and voltage RMSE metrics."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--simulation-run-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit simulation run IDs to compare.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Optional cell filter (e.g., CELL_A CELL_B).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model filter (SPM SPMe DFN).",
    )
    parser.add_argument(
        "--parameter-set",
        default=None,
        help="Optional base parameter-set filter (e.g., Chen2020).",
    )
    parser.add_argument(
        "--name-extention",
        default=None,
        help="Optional parameter-set name_extention filter.",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="Optional run_name filter.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Keep latest N rows after filtering (ordered by simulation_run_id desc).",
    )
    parser.add_argument(
        "--cycle",
        type=str,
        default=None,
        help="Optional cycle index or consecutive cycle range (e.g., 2 or 1-2).",
    )
    parser.add_argument(
        "--capacity-tail-stride",
        type=int,
        default=None,
        help=(
            "Optional common tail stride for capacity-window RMSE (>=120s per step). "
            "If omitted: optimized runs use stage2 stride from optimization config, "
            "non-optimization runs use 1."
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=["created_at", "objective", "full_rmse"],
        default="objective",
        help="Sort key for comparison table.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (default is ascending).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--print-all-columns",
        action="store_true",
        help="Print full table (may be wide due to dynamic parameter columns).",
    )
    return parser.parse_args()


def _parse_json_object(raw: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return {str(k): v for k, v in parsed.items()}
    return {}


def _coerce_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def parse_cycle_range(cycle_arg: str | None) -> tuple[int, int] | None:
    if cycle_arg is None:
        return None
    raw = cycle_arg.strip()
    if not raw:
        return None
    if "-" in raw:
        left, right = raw.split("-", 1)
        start = int(left)
        end = int(right)
    else:
        start = int(raw)
        end = start
    if start <= 0 or end <= 0:
        raise ValueError("--cycle must be positive (e.g., 2 or 1-2).")
    if end < start:
        raise ValueError("--cycle range must be ascending (e.g., 1-2).")
    return (start, end)


def _rmse_for_mask(error_v: np.ndarray, mask: np.ndarray) -> float:
    vals = error_v[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(vals))))


def _capacity_tail_mask(
    segment_id: np.ndarray,
    time_in_step_s: np.ndarray,
    tail_stride: int,
) -> np.ndarray:
    base_mask = time_in_step_s >= 120.0
    if int(tail_stride) <= 1:
        return base_mask
    keep = np.zeros_like(base_mask, dtype=bool)
    for sid in np.unique(segment_id[base_mask]):
        idx = np.flatnonzero((segment_id == sid) & base_mask)
        if idx.size == 0:
            continue
        picked = idx[:: int(tail_stride)]
        if picked.size == 0 or picked[-1] != idx[-1]:
            picked = np.append(picked, idx[-1])
        keep[picked] = True
    return keep


def compute_window_rmses_from_aligned(
    aligned: pd.DataFrame,
    capacity_tail_stride: int,
) -> dict[str, float]:
    if aligned.empty:
        return {k: float("nan") for k in RMSE_COLUMNS}

    cycle = aligned["cycle_index"].to_numpy(dtype=float)
    step = aligned["step_index"].to_numpy(dtype=float)
    t = aligned["test_time_s"].to_numpy(dtype=float)
    error_v = aligned["voltage_sim_v"].to_numpy(dtype=float) - aligned["voltage_exp_v"].to_numpy(
        dtype=float
    )

    cycle_changed = np.ones_like(cycle, dtype=bool)
    cycle_changed[1:] = cycle[1:] != cycle[:-1]
    step_changed = np.ones_like(step, dtype=bool)
    step_changed[1:] = step[1:] != step[:-1]
    segment_id = np.cumsum(cycle_changed | step_changed)

    time_in_step = np.zeros_like(t)
    for sid in np.unique(segment_id):
        idx = np.flatnonzero(segment_id == sid)
        if idx.size == 0:
            continue
        time_in_step[idx] = t[idx] - t[idx[0]]

    capacity_mask = _capacity_tail_mask(
        segment_id=segment_id,
        time_in_step_s=time_in_step,
        tail_stride=int(capacity_tail_stride),
    )

    return {
        "rmse_ohmic_0_2s_V": _rmse_for_mask(error_v, (time_in_step >= 0.0) & (time_in_step < 2.0)),
        "rmse_kinetic_2_20s_V": _rmse_for_mask(error_v, (time_in_step >= 2.0) & (time_in_step < 20.0)),
        "rmse_diffusion_20_120s_V": _rmse_for_mask(
            error_v, (time_in_step >= 20.0) & (time_in_step < 120.0)
        ),
        "rmse_capacity_120plus_s_V": _rmse_for_mask(error_v, capacity_mask),
        "rmse_full_profile_V": float(np.sqrt(np.mean(np.square(error_v)))),
    }


def _build_where_clause(args: argparse.Namespace) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []

    if args.simulation_run_ids:
        placeholders = ",".join(["?"] * len(args.simulation_run_ids))
        clauses.append(f"sr.id IN ({placeholders})")
        params.extend([int(v) for v in args.simulation_run_ids])
    if args.cells:
        placeholders = ",".join(["?"] * len(args.cells))
        clauses.append(f"c.cell_code IN ({placeholders})")
        params.extend([str(v) for v in args.cells])
    if args.models:
        placeholders = ",".join(["?"] * len(args.models))
        clauses.append(f"sr.model_name IN ({placeholders})")
        params.extend([str(v) for v in args.models])
    if args.parameter_set is not None:
        clauses.append("ps.base_parameter_set_name = ?")
        params.append(str(args.parameter_set))
    if args.name_extention is not None:
        clauses.append("ps.name_extention = ?")
        params.append(str(args.name_extention))
    if args.run_names:
        placeholders = ",".join(["?"] * len(args.run_names))
        clauses.append(f"sr.run_name IN ({placeholders})")
        params.extend([str(v) for v in args.run_names])

    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def load_candidate_runs(args: argparse.Namespace) -> list[sqlite3.Row]:
    if not args.db_path.exists():
        raise FileNotFoundError(f"DB not found: {args.db_path}")

    where_clause, params = _build_where_clause(args)
    query = f"""
        SELECT
            sr.id AS simulation_run_id,
            c.cell_code,
            sr.model_name,
            ps.base_parameter_set_name,
            ps.name_extention,
            sr.run_name,
            sr.capacity_ah,
            sr.initial_soc,
            sr.parameter_overrides_json,
            oru.id AS optimization_id,
            oru.base_simulation_run_id,
            oru.created_at_ts_utc,
            oru.objective_v,
            oru.optimization_config_json,
            oru.best_result_json
        FROM simulation_runs sr
        JOIN cells c ON c.id = sr.cell_id
        JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
        LEFT JOIN optimization_runs oru ON oru.simulation_run_id = sr.id
        {where_clause}
        ORDER BY sr.id DESC;
    """
    with sqlite3.connect(args.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    if args.latest is not None:
        if int(args.latest) <= 0:
            raise ValueError("--latest must be >= 1.")
        rows = rows[: int(args.latest)]
    return rows


def _load_latest_experimental_run_id(conn: sqlite3.Connection, cell_code: str) -> int:
    row = conn.execute(
        """
        SELECT er.id
        FROM experimental_runs er
        JOIN cells c ON c.id = er.cell_id
        WHERE c.cell_code = ?
        ORDER BY er.id DESC
        LIMIT 1;
        """,
        (cell_code,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No experimental run found for cell={cell_code}")
    return int(row[0])


def _load_curve(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    run_id: int,
    cycle_range: tuple[int, int] | None,
) -> pd.DataFrame:
    if cycle_range is None:
        rows = conn.execute(
            f"""
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM {table}
            WHERE {id_column} = ?
            ORDER BY test_time_s ASC;
            """,
            (int(run_id),),
        ).fetchall()
    elif cycle_range[0] == cycle_range[1]:
        rows = conn.execute(
            f"""
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM {table}
            WHERE {id_column} = ? AND cycle_index = ?
            ORDER BY test_time_s ASC;
            """,
            (int(run_id), int(cycle_range[0])),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM {table}
            WHERE {id_column} = ? AND cycle_index BETWEEN ? AND ?
            ORDER BY test_time_s ASC;
            """,
            (int(run_id), int(cycle_range[0]), int(cycle_range[1])),
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["test_time_s", "cycle_index", "step_index", "current_a", "voltage_v"],
    )


def _build_aligned_trace(exp_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
    exp = exp_df.dropna(subset=["test_time_s", "cycle_index", "step_index", "voltage_v"]).copy()
    sim = sim_df.dropna(subset=["test_time_s", "voltage_v"]).copy()
    if exp.empty or sim.empty:
        return pd.DataFrame(
            columns=["test_time_s", "cycle_index", "step_index", "voltage_exp_v", "voltage_sim_v"]
        )

    exp_t = exp["test_time_s"].to_numpy(dtype=float)
    sim_t = sim["test_time_s"].to_numpy(dtype=float)
    sim_v = sim["voltage_v"].to_numpy(dtype=float)
    sim_v_interp = np.interp(exp_t, sim_t, sim_v, left=np.nan, right=np.nan)

    aligned = pd.DataFrame(
        {
            "test_time_s": exp_t,
            "cycle_index": exp["cycle_index"].to_numpy(dtype=float),
            "step_index": exp["step_index"].to_numpy(dtype=float),
            "voltage_exp_v": exp["voltage_v"].to_numpy(dtype=float),
            "voltage_sim_v": sim_v_interp,
        }
    )
    return aligned.dropna(subset=["voltage_exp_v", "voltage_sim_v"]).reset_index(drop=True)


def build_comparison_dataframe(
    db_path: Path,
    rows: list[sqlite3.Row],
    cycle_range: tuple[int, int] | None,
    capacity_tail_stride_override: int | None,
) -> pd.DataFrame:
    flattened: list[dict[str, object]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        exp_cache: dict[str, pd.DataFrame] = {}
        for row in rows:
            best = _parse_json_object(row["best_result_json"])
            config = _parse_json_object(row["optimization_config_json"])
            parameter_scales = best.get("parameter_scales", {})
            if not isinstance(parameter_scales, dict):
                parameter_scales = {}
            variables = best.get("variables", [])
            if not isinstance(variables, list):
                variables = []

            cell_code = str(row["cell_code"])
            if cell_code not in exp_cache:
                exp_run_id = _load_latest_experimental_run_id(conn, cell_code=cell_code)
                exp_cache[cell_code] = _load_curve(
                    conn,
                    table="experimental_timeseries",
                    id_column="experimental_run_id",
                    run_id=exp_run_id,
                    cycle_range=cycle_range,
                )
            exp_df = exp_cache[cell_code]
            sim_df = _load_curve(
                conn,
                table="simulation_timeseries",
                id_column="simulation_run_id",
                run_id=int(row["simulation_run_id"]),
                cycle_range=cycle_range,
            )
            aligned = _build_aligned_trace(exp_df=exp_df, sim_df=sim_df)

            if capacity_tail_stride_override is not None:
                tail_stride = int(capacity_tail_stride_override)
            else:
                tail_stride = int(config.get("tail_downsample_stride_stage2", 1))
            tail_stride = max(1, int(tail_stride))
            rmses = compute_window_rmses_from_aligned(
                aligned=aligned,
                capacity_tail_stride=tail_stride,
            )

            is_optimized = row["optimization_id"] is not None
            objective_default = rmses["rmse_full_profile_V"]
            objective_stored = best.get("objective_stage2_fit_V", row["objective_v"])
            item: dict[str, object] = {
                "optimization_id": (
                    int(row["optimization_id"]) if row["optimization_id"] is not None else None
                ),
                "simulation_run_id": int(row["simulation_run_id"]),
                "base_simulation_run_id": row["base_simulation_run_id"],
                "created_at_ts_utc": str(
                    row["created_at_ts_utc"]
                    if row["created_at_ts_utc"] is not None
                    else f"sim_run_{int(row['simulation_run_id'])}"
                ),
                "cell_code": cell_code,
                "model_name": str(row["model_name"]),
                "base_parameter_set_name": str(row["base_parameter_set_name"]),
                "name_extention": str(row["name_extention"]),
                "run_name": row["run_name"],
                "run_source": "optimization" if is_optimized else "simulation_only",
                "is_basecase_name_extention": int(str(row["name_extention"]) == ""),
                "objective_v": _coerce_float(
                    objective_stored if is_optimized else objective_default
                ),
                "capacity_ah": _coerce_float(
                    best.get("capacity_ah", row["capacity_ah"])
                ),
                "initial_soc": _coerce_float(
                    best.get("initial_soc", row["initial_soc"])
                ),
                "cycle": (
                    config.get("cycle", None)
                    if cycle_range is None
                    else (
                        cycle_range[0] if cycle_range[0] == cycle_range[1] else f"{cycle_range[0]}-{cycle_range[1]}"
                    )
                ),
                "capacity_tail_stride_used": tail_stride,
            }

            for rmse_key in RMSE_COLUMNS:
                item[rmse_key] = _coerce_float(rmses.get(rmse_key))

            for key, value in parameter_scales.items():
                item[f"scale::{str(key)}"] = _coerce_float(value)

            for var in variables:
                if not isinstance(var, dict):
                    continue
                name = str(var.get("name", "")).strip()
                if not name:
                    continue
                value_key = f"var_{name}"
                if value_key in best:
                    item[f"var::{name}"] = _coerce_float(best.get(value_key))
                parameter_name = var.get("parameter_name", None)
                if parameter_name is not None:
                    item[f"var_parameter::{name}"] = str(parameter_name)

            flattened.append(item)

    if not flattened:
        return pd.DataFrame()
    return pd.DataFrame(flattened)


def sort_dataframe(df: pd.DataFrame, sort_by: str, descending: bool) -> pd.DataFrame:
    if sort_by == "created_at":
        return df.sort_values("created_at_ts_utc", ascending=not descending).reset_index(drop=True)
    if sort_by == "full_rmse":
        return df.sort_values("rmse_full_profile_V", ascending=not descending).reset_index(drop=True)
    return df.sort_values("objective_v", ascending=not descending).reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    n_opt = int((df["run_source"] == "optimization").sum()) if "run_source" in df.columns else 0
    n_base = int((df["run_source"] == "simulation_only").sum()) if "run_source" in df.columns else 0
    print(f"Selected runs: total={len(df)}, optimization={n_opt}, simulation_only={n_base}")
    best_idx = int(df["objective_v"].astype(float).idxmin())
    best = df.loc[best_idx]
    print(
        "Best objective run: "
        f"simulation_run_id={int(best['simulation_run_id'])}, "
        f"run_name={best.get('run_name', None)}, "
        f"objective_v={float(best['objective_v']):.6f}, "
        f"full_rmse={float(best['rmse_full_profile_V']):.6f}"
    )


def print_table(df: pd.DataFrame, print_all_columns: bool) -> None:
    leading = [
        "simulation_run_id",
        "run_source",
        "run_name",
        "cell_code",
        "model_name",
        "base_parameter_set_name",
        "name_extention",
        "is_basecase_name_extention",
        "cycle",
        "capacity_ah",
        "initial_soc",
        "capacity_tail_stride_used",
        "objective_v",
        *RMSE_COLUMNS,
    ]
    existing_leading = [c for c in leading if c in df.columns]
    dynamic_columns = [
        c
        for c in df.columns
        if c not in existing_leading
        and (c.startswith("var::") or c.startswith("scale::") or c.startswith("var_parameter::"))
    ]
    view_columns = existing_leading + sorted(dynamic_columns)
    view = df[view_columns].copy()
    if not print_all_columns:
        pd.set_option("display.max_columns", 40)
        pd.set_option("display.width", 220)
    print(view.to_string(index=False))


def main() -> None:
    args = parse_args()
    cycle_range = parse_cycle_range(args.cycle)
    if args.capacity_tail_stride is not None and int(args.capacity_tail_stride) <= 0:
        raise ValueError("--capacity-tail-stride must be >= 1.")

    rows = load_candidate_runs(args)
    if not rows:
        print("No matching simulation runs found.")
        return

    df = build_comparison_dataframe(
        db_path=args.db_path,
        rows=rows,
        cycle_range=cycle_range,
        capacity_tail_stride_override=(
            None if args.capacity_tail_stride is None else int(args.capacity_tail_stride)
        ),
    )
    if df.empty:
        print("No comparable rows could be parsed from optimization metadata.")
        return

    df = sort_dataframe(df, sort_by=str(args.sort_by), descending=bool(args.descending))
    print_summary(df)
    print_table(df, print_all_columns=bool(args.print_all_columns))

    output_csv = args.output_csv
    if output_csv is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_csv = DEFAULT_OUTPUT_DIR / "phase5_optimization_comparison.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved comparison CSV: {output_csv}")


if __name__ == "__main__":
    main()
