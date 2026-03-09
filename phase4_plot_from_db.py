from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DB_PATH = Path("outputs/phase4/battery_pipeline.db")
DEFAULT_OUTPUT_DIR = Path("outputs/phase4")


@dataclass(frozen=True)
class SimulationRunMeta:
    simulation_run_id: int
    cell_code: str
    model_name: str
    base_parameter_set_name: str
    name_extention: str
    run_name: str | None
    capacity_ah: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4 decoupled visualization: read experiment and simulation curves "
            "from SQLite and plot voltage comparison."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of only saving PNG.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to populated SQLite DB.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG output(s) are saved.",
    )
    parser.add_argument(
        "--simulation-run-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit simulation_run.id list to plot together.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Cell code filters (e.g., CELL_A CELL_B).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["SPM", "SPMe", "DFN"],
        default=None,
        help="Model name filters (multi-value).",
    )
    parser.add_argument(
        "--parameter-sets",
        nargs="+",
        default=None,
        help="Base parameter set filters (multi-value, e.g., Chen2020).",
    )
    parser.add_argument(
        "--parameter-name-extention",
        default=None,
        help="Parameter suffix filter (e.g., _optimized_v1).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Simulation run_name filter.",
    )
    parser.add_argument(
        "--default-parameters-only",
        action="store_true",
        help="Treat default parameters as name_extention=''.",
    )
    parser.add_argument(
        "--cycle",
        type=str,
        default=None,
        help="Optional cycle index or consecutive cycle range (e.g., 2 or 1-2).",
    )
    parser.add_argument(
        "--plot-with-current",
        action="store_true",
        help="Add a current subplot above the voltage comparison plot.",
    )
    parser.add_argument(
        "--series-mode",
        choices=["both", "experiment-only", "simulation-only"],
        default="both",
        help="Choose which curves to plot.",
    )
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="List parameter sets used by matching simulation runs and exit.",
    )
    parser.add_argument(
        "--show-default-parameter-set",
        action="store_true",
        help="Print default parameter set (base + overridden) for matching/all cells.",
    )
    parser.add_argument(
        "--list-optimization-runs",
        action="store_true",
        help=(
            "List optimization metadata for matching simulation runs (from "
            "optimization_runs table) and exit."
        ),
    )
    return parser.parse_args()


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def resolve_simulation_runs(
    conn: sqlite3.Connection, args: argparse.Namespace
) -> list[SimulationRunMeta]:
    selected_run_ids: list[int] = []
    if args.simulation_run_ids is not None:
        selected_run_ids.extend(int(x) for x in args.simulation_run_ids)
    selected_run_ids = sorted(set(selected_run_ids))

    selected_models = args.models
    selected_parameter_sets = args.parameter_sets

    if selected_run_ids:
        placeholders = ",".join(["?"] * len(selected_run_ids))
        rows = conn.execute(
            f"""
            SELECT
                sr.id,
                c.cell_code,
                sr.model_name,
                ps.base_parameter_set_name,
                ps.name_extention,
                sr.run_name,
                sr.capacity_ah
            FROM simulation_runs sr
            JOIN cells c ON c.id = sr.cell_id
            JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
            WHERE sr.id IN ({placeholders})
            ORDER BY sr.id ASC;
            """,
            selected_run_ids,
        ).fetchall()
    else:
        where = []
        params: list[object] = []
        if args.cells is not None:
            placeholders = ",".join(["?"] * len(args.cells))
            where.append(f"c.cell_code IN ({placeholders})")
            params.extend(args.cells)
        if selected_models is not None:
            placeholders = ",".join(["?"] * len(selected_models))
            where.append(f"sr.model_name IN ({placeholders})")
            params.extend(selected_models)
        if selected_parameter_sets is not None:
            placeholders = ",".join(["?"] * len(selected_parameter_sets))
            where.append(f"ps.base_parameter_set_name IN ({placeholders})")
            params.extend(selected_parameter_sets)
        if args.parameter_name_extention is not None:
            where.append("ps.name_extention = ?")
            params.append(args.parameter_name_extention)
        if args.default_parameters_only:
            where.append("ps.name_extention = ''")
        if args.run_name is not None:
            where.append("sr.run_name = ?")
            params.append(args.run_name)

        where_sql = " AND ".join(where)
        if where_sql:
            where_sql = "WHERE " + where_sql

        rows = conn.execute(
            f"""
            SELECT
                sr.id,
                c.cell_code,
                sr.model_name,
                ps.base_parameter_set_name,
                ps.name_extention,
                sr.run_name,
                sr.capacity_ah
            FROM simulation_runs sr
            JOIN cells c ON c.id = sr.cell_id
            JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
            {where_sql}
            ORDER BY sr.id ASC;
            """,
            params,
        ).fetchall()

    result = [
        SimulationRunMeta(
            simulation_run_id=int(row[0]),
            cell_code=str(row[1]),
            model_name=str(row[2]),
            base_parameter_set_name=str(row[3]),
            name_extention=str(row[4]),
            run_name=row[5],
            capacity_ah=float(row[6]) if row[6] is not None else None,
        )
        for row in rows
    ]
    if not result:
        raise RuntimeError("No simulation runs matched the provided filters.")
    return result


def resolve_experimental_cells(
    conn: sqlite3.Connection, requested_cells: list[str] | None
) -> list[str]:
    if requested_cells:
        return sorted(set(requested_cells))
    rows = conn.execute(
        """
        SELECT DISTINCT c.cell_code
        FROM cells c
        JOIN experimental_runs er ON er.cell_id = c.id
        ORDER BY c.cell_code ASC;
        """
    ).fetchall()
    cells = [str(row[0]) for row in rows]
    if not cells:
        raise RuntimeError("No experimental runs found in database.")
    return cells


def list_parameter_sets_used_by_runs(
    conn: sqlite3.Connection,
    runs: list[SimulationRunMeta],
) -> None:
    run_ids = sorted({run.simulation_run_id for run in runs})
    if not run_ids:
        print("No matching simulation runs.")
        return
    placeholders = ",".join(["?"] * len(run_ids))
    rows = conn.execute(
        f"""
        SELECT DISTINCT
            NULL AS cell_code,
            ps.base_parameter_set_name,
            ps.name_extention,
            ps.modified_parameters_json,
            ps.base_parameters_json
        FROM simulation_runs sr
        JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
        WHERE sr.id IN ({placeholders})
        ORDER BY ps.base_parameter_set_name, ps.name_extention, cell_code;
        """,
        run_ids,
    ).fetchall()
    print_parameter_set_rows("Parameter sets used by selected runs:", rows)

    run_rows = conn.execute(
        f"""
        SELECT
            sr.id,
            c.cell_code,
            sr.model_name,
            COALESCE(sr.parameter_overrides_json, '{{}}')
        FROM simulation_runs sr
        JOIN cells c ON c.id = sr.cell_id
        WHERE sr.id IN ({placeholders})
        ORDER BY sr.id ASC;
        """,
        run_ids,
    ).fetchall()
    print("Run-level effective overrides (includes capacity/parallel scaling):")
    for run_id, cell_code, model_name, overrides_json in run_rows:
        print(f"- run_id={run_id}, cell={cell_code}, model={model_name}")
        try:
            parsed = json.loads(str(overrides_json or "{}"))
        except Exception:
            print(f"  overrides={overrides_json}")
            continue
        if not isinstance(parsed, dict) or not parsed:
            print("  overrides={}")
            continue
        for key in sorted(parsed.keys()):
            print(f"  - {key}: {parsed[key]}")


def print_parameter_set_rows(
    title: str,
    rows: list[tuple[object, object, object, object, object]],
) -> None:
    print(title)
    for cell_code, base_name, name_ext, overridden_json, base_json in rows:
        suffix = str(name_ext or "")
        cell_prefix = f"{cell_code}: " if cell_code is not None else ""
        try:
            base_param_count = len(json.loads(str(base_json or "{}")))
        except Exception:
            base_param_count = -1
        print(
            f"- {cell_prefix}base={base_name}, "
            f"name_extention='{suffix}', overridden={overridden_json}"
        )
        if base_param_count < 0:
            print(f"  base_parameters={base_json}")
            continue

        print(f"  base_parameter_count={base_param_count}")
        try:
            parsed_base = json.loads(str(base_json or "{}"))
        except Exception:
            print(f"  base_parameters={base_json}")
            continue
        if not isinstance(parsed_base, dict):
            print(f"  base_parameters={base_json}")
            continue

        print("  base_parameters:")
        for key in sorted(parsed_base.keys()):
            print(f"    - {key}: {parsed_base[key]}")


def show_default_parameter_sets(
    conn: sqlite3.Connection,
    requested_cells: list[str] | None,
) -> None:
    where_clause = ""
    params: list[object] = []
    if requested_cells:
        placeholders = ",".join(["?"] * len(requested_cells))
        where_clause = f"WHERE c.cell_code IN ({placeholders})"
        params.extend(requested_cells)
    rows = conn.execute(
        f"""
        SELECT
            c.cell_code,
            ps.base_parameter_set_name,
            ps.name_extention,
            ps.modified_parameters_json,
            ps.base_parameters_json
        FROM cells c
        LEFT JOIN parameter_sets ps ON ps.id = c.default_parameter_set_id
        {where_clause}
        ORDER BY c.cell_code ASC;
        """,
        params,
    ).fetchall()
    if not rows:
        print("No matching cells found for default parameter-set lookup.")
        return
    missing = [str(row[0]) for row in rows if row[1] is None]
    present_rows = [row for row in rows if row[1] is not None]
    if present_rows:
        print_parameter_set_rows("Default parameter set by cell:", present_rows)
    for cell_code in missing:
        print(f"- {cell_code}: <not set>")


def list_optimization_runs_for_selection(
    conn: sqlite3.Connection,
    runs: list[SimulationRunMeta],
) -> None:
    run_ids = sorted({run.simulation_run_id for run in runs})
    if not run_ids:
        print("No matching simulation runs.")
        return
    placeholders = ",".join(["?"] * len(run_ids))
    rows = conn.execute(
        f"""
        SELECT
            sr.id,
            c.cell_code,
            sr.model_name,
            ps.base_parameter_set_name,
            ps.name_extention,
            sr.run_name,
            oru.base_simulation_run_id,
            oru.created_at_ts_utc,
            oru.objective_v,
            oru.optimization_config_json
        FROM optimization_runs oru
        JOIN simulation_runs sr ON sr.id = oru.simulation_run_id
        JOIN cells c ON c.id = sr.cell_id
        JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
        WHERE sr.id IN ({placeholders})
        ORDER BY sr.id ASC;
        """,
        run_ids,
    ).fetchall()
    if not rows:
        print("No optimization metadata rows found for selected simulation runs.")
        return
    print("Optimization runs:")
    for row in rows:
        (
            run_id,
            cell_code,
            model_name,
            base_set,
            name_ext,
            run_name,
            base_run_id,
            created_at,
            objective_v,
            config_json,
        ) = row
        print(
            f"- run_id={run_id}, cell={cell_code}, model={model_name}, "
            f"base_set={base_set}, name_extention='{name_ext}', run_name={run_name}, "
            f"base_run_id={base_run_id}, objective_v={objective_v}, created_at={created_at}"
        )
        try:
            parsed = json.loads(str(config_json or "{}"))
        except Exception:
            print(f"  optimization_config_json={config_json}")
            continue
        if not isinstance(parsed, dict):
            print(f"  optimization_config_json={config_json}")
            continue
        cycle_value = parsed.get("cycle", None)
        requested_ext = parsed.get("db_parameter_name_extention_requested", None)
        resolved_ext = parsed.get("db_parameter_name_extention_resolved", None)
        print(
            f"  cycle={cycle_value}, ext_requested={requested_ext}, ext_resolved={resolved_ext}"
        )


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


def get_latest_experimental_run_id(conn: sqlite3.Connection, cell_code: str) -> int:
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
        raise RuntimeError(f"No experimental run found for {cell_code}")
    return int(row[0])


def load_experimental_curve(
    conn: sqlite3.Connection, experimental_run_id: int, cycle_range: tuple[int, int] | None
) -> pd.DataFrame:
    if cycle_range is None:
        rows = conn.execute(
            """
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM experimental_timeseries
            WHERE experimental_run_id = ?
            ORDER BY test_time_s ASC;
            """,
            (experimental_run_id,),
        ).fetchall()
    elif cycle_range[0] == cycle_range[1]:
        rows = conn.execute(
            """
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM experimental_timeseries
            WHERE experimental_run_id = ? AND cycle_index = ?
            ORDER BY test_time_s ASC;
            """,
            (experimental_run_id, cycle_range[0]),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM experimental_timeseries
            WHERE experimental_run_id = ? AND cycle_index BETWEEN ? AND ?
            ORDER BY test_time_s ASC;
            """,
            (experimental_run_id, cycle_range[0], cycle_range[1]),
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["test_time_s", "cycle_index", "step_index", "current_a", "voltage_v"],
    )


def load_simulation_curve(
    conn: sqlite3.Connection, simulation_run_id: int, cycle_range: tuple[int, int] | None
) -> pd.DataFrame:
    if cycle_range is None:
        rows = conn.execute(
            """
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM simulation_timeseries
            WHERE simulation_run_id = ?
            ORDER BY test_time_s ASC;
            """,
            (simulation_run_id,),
        ).fetchall()
    elif cycle_range[0] == cycle_range[1]:
        rows = conn.execute(
            """
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM simulation_timeseries
            WHERE simulation_run_id = ? AND cycle_index = ?
            ORDER BY test_time_s ASC;
            """,
            (simulation_run_id, cycle_range[0]),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT test_time_s, cycle_index, step_index, current_a, voltage_v
            FROM simulation_timeseries
            WHERE simulation_run_id = ? AND cycle_index BETWEEN ? AND ?
            ORDER BY test_time_s ASC;
            """,
            (simulation_run_id, cycle_range[0], cycle_range[1]),
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["test_time_s", "cycle_index", "step_index", "current_a", "voltage_v"],
    )


def build_aligned_dataframe(
    exp_df: pd.DataFrame, sim_df: pd.DataFrame
) -> pd.DataFrame:
    exp_df = exp_df.dropna(subset=["test_time_s", "voltage_v"]).copy()
    sim_df = sim_df.dropna(subset=["test_time_s", "voltage_v"]).copy()
    if exp_df.empty or sim_df.empty:
        raise RuntimeError("Missing experimental or simulation points after NA filtering.")

    exp_t = exp_df["test_time_s"].to_numpy(dtype=float)
    sim_t = sim_df["test_time_s"].to_numpy(dtype=float)

    sim_v = sim_df["voltage_v"].to_numpy(dtype=float)
    sim_v_interp = np.interp(exp_t, sim_t, sim_v, left=np.nan, right=np.nan)

    exp_i = exp_df["current_a"].to_numpy(dtype=float)
    sim_i = sim_df["current_a"].to_numpy(dtype=float)
    sim_i_interp = np.interp(exp_t, sim_t, sim_i, left=np.nan, right=np.nan)

    aligned = pd.DataFrame(
        {
            "test_time_s": exp_t,
            "voltage_exp_v": exp_df["voltage_v"].to_numpy(dtype=float),
            "voltage_sim_v": sim_v_interp,
            "current_exp_a": exp_i,
            "current_sim_a": sim_i_interp,
        }
    )
    return aligned.dropna(subset=["voltage_exp_v", "voltage_sim_v"])


def build_output_path(
    output_dir: Path,
    has_current: bool,
    run_ids: list[int],
    cell_codes: list[str],
    cycle_range: tuple[int, int] | None,
) -> Path:
    suffix = "_voltage_compare_with_current" if has_current else "_voltage_compare"
    run_ids_part = "-".join(str(rid) for rid in sorted(run_ids)) if run_ids else "exp_only"
    cells_part = "-".join(sorted(cell_codes))
    cycle_part = ""
    if cycle_range is not None:
        if cycle_range[0] == cycle_range[1]:
            cycle_part = f"_cycle_{cycle_range[0]}"
        else:
            cycle_part = f"_cycles_{cycle_range[0]}-{cycle_range[1]}"
    filename = f"cells_{cells_part}_runs_{run_ids_part}{cycle_part}{suffix}.png"
    return output_dir / filename


def plot_comparison(
    aligned_series: list[tuple[SimulationRunMeta, pd.DataFrame]],
    exp_by_cell: dict[str, pd.DataFrame],
    cycle_range: tuple[int, int] | None,
    output_path: Path | None,
    plot_current_above_voltage: bool,
    series_mode: str,
    show: bool,
) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_from_sim = {meta.cell_code for meta, _ in aligned_series}
    cells_sorted = sorted(set(exp_by_cell.keys()) | cells_from_sim)
    if not cells_sorted:
        raise RuntimeError("No cells found to plot.")
    aligned_by_cell: dict[str, list[tuple[SimulationRunMeta, pd.DataFrame]]] = {
        cell_code: [] for cell_code in cells_sorted
    }
    for meta, aligned_df in aligned_series:
        aligned_by_cell.setdefault(meta.cell_code, []).append((meta, aligned_df))

    if cycle_range is None:
        cycle_label = "all cycles"
    elif cycle_range[0] == cycle_range[1]:
        cycle_label = f"cycle {cycle_range[0]}"
    else:
        cycle_label = f"cycles {cycle_range[0]}-{cycle_range[1]}"

    n_cells = len(cells_sorted)
    if plot_current_above_voltage:
        fig, axes = plt.subplots(
            2,
            n_cells,
            figsize=(5.5 * n_cells, 7.0),
            sharex=False,
            squeeze=False,
            gridspec_kw={"height_ratios": [1, 2]},
        )
        for idx, cell_code in enumerate(cells_sorted):
            ax_i = axes[0, idx]
            ax_v = axes[1, idx]
            exp_df = exp_by_cell.get(
                cell_code,
                pd.DataFrame(columns=["test_time_s", "current_a", "voltage_v"]),
            )
            exp_i_df = exp_df.dropna(subset=["test_time_s", "current_a"])
            exp_v_df = exp_df.dropna(subset=["test_time_s", "voltage_v"])

            if series_mode in ("both", "experiment-only"):
                ax_i.plot(
                    exp_i_df["test_time_s"].to_numpy(dtype=float) / 3600.0,
                    exp_i_df["current_a"],
                    label=f"Exp current {cell_code}",
                    linewidth=1.4,
                )
            ax_i_c = ax_i.twinx()
            c_rate_any = False
            if series_mode in ("both", "simulation-only"):
                for meta, aligned_df in aligned_by_cell.get(cell_code, []):
                    sim_label = (
                        f"Sim current run{meta.simulation_run_id} "
                        f"({meta.model_name}, {meta.base_parameter_set_name}{meta.name_extention})"
                    )
                    x_h = aligned_df["test_time_s"].to_numpy(dtype=float) / 3600.0
                    ax_i.plot(
                        x_h,
                        aligned_df["current_sim_a"],
                        label=sim_label,
                        linewidth=1.1,
                        linestyle="--",
                    )
                    if meta.capacity_ah is not None and meta.capacity_ah > 0:
                        c_rate_any = True
                        ax_i_c.plot(
                            x_h,
                            aligned_df["current_sim_a"] / meta.capacity_ah,
                            linewidth=1.0,
                            linestyle=":",
                            alpha=0.75,
                            label=f"C-rate run{meta.simulation_run_id}",
                        )
            ax_i.set_ylabel("Current [A]")
            ax_i.set_title(f"{cell_code} ({cycle_label})")
            ax_i.grid(True, alpha=0.3)
            ax_i.legend(fontsize=8, loc="upper left")
            if c_rate_any:
                ax_i_c.set_ylabel("C-rate [1/h]")
                ax_i_c.legend(fontsize=8, loc="upper right")

            if series_mode in ("both", "experiment-only"):
                ax_v.plot(
                    exp_v_df["test_time_s"].to_numpy(dtype=float) / 3600.0,
                    exp_v_df["voltage_v"],
                    label=f"Exp voltage {cell_code}",
                    linewidth=1.6,
                )
            if series_mode in ("both", "simulation-only"):
                for meta, aligned_df in aligned_by_cell.get(cell_code, []):
                    sim_label = (
                        f"Sim voltage run{meta.simulation_run_id} "
                        f"({meta.model_name}, {meta.base_parameter_set_name}{meta.name_extention})"
                    )
                    ax_v.plot(
                        aligned_df["test_time_s"].to_numpy(dtype=float) / 3600.0,
                        aligned_df["voltage_sim_v"],
                        label=sim_label,
                        linewidth=1.4,
                        linestyle="--",
                    )
            ax_v.set_xlabel("Time [h]")
            ax_v.set_ylabel("Voltage [V]")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend(fontsize=8)
    else:
        fig, axes = plt.subplots(
            1,
            n_cells,
            figsize=(5.5 * n_cells, 4.2),
            sharex=False,
            squeeze=False,
        )
        for idx, cell_code in enumerate(cells_sorted):
            ax_v = axes[0, idx]
            exp_df = exp_by_cell.get(
                cell_code,
                pd.DataFrame(columns=["test_time_s", "voltage_v"]),
            )
            exp_v_df = exp_df.dropna(subset=["test_time_s", "voltage_v"])
            if series_mode in ("both", "experiment-only"):
                ax_v.plot(
                    exp_v_df["test_time_s"].to_numpy(dtype=float) / 3600.0,
                    exp_v_df["voltage_v"],
                    label=f"Exp voltage {cell_code}",
                    linewidth=1.6,
                )
            if series_mode in ("both", "simulation-only"):
                for meta, aligned_df in aligned_by_cell.get(cell_code, []):
                    sim_label = (
                        f"Sim voltage run{meta.simulation_run_id} "
                        f"({meta.model_name}, {meta.base_parameter_set_name}{meta.name_extention})"
                    )
                    ax_v.plot(
                        aligned_df["test_time_s"].to_numpy(dtype=float) / 3600.0,
                        aligned_df["voltage_sim_v"],
                        label=sim_label,
                        linewidth=1.4,
                        linestyle="--",
                    )
            ax_v.set_title(f"{cell_code} ({cycle_label})")
            ax_v.set_xlabel("Time [h]")
            ax_v.set_ylabel("Voltage [V]")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend(fontsize=8)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)

def main() -> None:
    args = parse_args()
    cycle_range = parse_cycle_range(args.cycle)
    if args.default_parameters_only and args.parameter_name_extention not in (None, ""):
        raise ValueError(
            "Use either --default-parameters-only or --parameter-name-extention, not both."
        )

    with connect_db(args.db_path) as conn:
        if args.show_default_parameter_set:
            show_default_parameter_sets(conn, args.cells)

        if (
            args.series_mode in ("both", "simulation-only")
            or args.list_parameters
            or args.list_optimization_runs
        ):
            runs = resolve_simulation_runs(conn, args)
            if args.list_parameters:
                list_parameter_sets_used_by_runs(conn, runs)
                return
            if args.list_optimization_runs:
                try:
                    list_optimization_runs_for_selection(conn, runs)
                except sqlite3.OperationalError:
                    print(
                        "optimization_runs table is not present in this DB yet. "
                        "Run phase5_run_optimization.py at least once to record metadata."
                    )
                return
        else:
            runs = []

        runs_by_cell: dict[str, list[SimulationRunMeta]] = {}
        for meta in runs:
            runs_by_cell.setdefault(meta.cell_code, []).append(meta)

        exp_by_cell: dict[str, pd.DataFrame] = {}
        aligned_series: list[tuple[SimulationRunMeta, pd.DataFrame]] = []

        if args.series_mode in ("both", "experiment-only"):
            if args.series_mode == "experiment-only":
                target_cells = resolve_experimental_cells(conn, args.cells)
            else:
                target_cells = sorted(runs_by_cell.keys())
            for cell_code in target_cells:
                exp_run_id = get_latest_experimental_run_id(conn, cell_code)
                exp_df = load_experimental_curve(conn, exp_run_id, cycle_range)
                if exp_df.empty:
                    raise RuntimeError(
                        f"No experimental points found for {cell_code}"
                        f"{' cycle(s) ' + args.cycle if args.cycle is not None else ''}."
                    )
                exp_by_cell[cell_code] = exp_df

        if args.series_mode in ("both", "simulation-only"):
            for cell_code, cell_runs in sorted(runs_by_cell.items()):
                exp_df = exp_by_cell.get(cell_code)
                for meta in cell_runs:
                    sim_df = load_simulation_curve(conn, meta.simulation_run_id, cycle_range)
                    if sim_df.empty:
                        raise RuntimeError(
                            f"No simulation points found for run_id={meta.simulation_run_id}"
                            f"{' cycle(s) ' + args.cycle if args.cycle is not None else ''}."
                        )
                    if args.series_mode == "both":
                        if exp_df is None:
                            raise RuntimeError(
                                f"Missing experimental data for {cell_code} while series-mode=both."
                            )
                        aligned_df = build_aligned_dataframe(exp_df=exp_df, sim_df=sim_df)
                    else:
                        aligned_df = pd.DataFrame(
                            {
                                "test_time_s": sim_df["test_time_s"].to_numpy(dtype=float),
                                "current_sim_a": sim_df["current_a"].to_numpy(dtype=float),
                                "voltage_sim_v": sim_df["voltage_v"].to_numpy(dtype=float),
                            }
                        ).dropna(subset=["test_time_s", "voltage_sim_v"])
                    aligned_series.append((meta, aligned_df))

        cell_codes_for_output = sorted(
            set(exp_by_cell.keys()) | {meta.cell_code for meta, _ in aligned_series}
        )
        run_ids_for_output = [m.simulation_run_id for m in runs]
        out_path = None if args.show else build_output_path(
            output_dir=args.output_dir,
            has_current=args.plot_with_current,
            run_ids=run_ids_for_output,
            cell_codes=cell_codes_for_output,
            cycle_range=cycle_range,
        )

        plot_comparison(
            aligned_series=aligned_series,
            exp_by_cell=exp_by_cell,
            cycle_range=cycle_range,
            output_path=out_path,
            plot_current_above_voltage=args.plot_with_current,
            series_mode=args.series_mode,
            show=args.show,
        )

        if args.show:
            print(
                f"Displayed plot "
                f"(mode={args.series_mode}, cells={cell_codes_for_output}, "
                f"runs={[m.simulation_run_id for m in runs]})"
            )
        else:
            print(
                f"Saved plot: {out_path} "
                f"(mode={args.series_mode}, cells={cell_codes_for_output}, "
                f"runs={[m.simulation_run_id for m in runs]})"
            )

if __name__ == "__main__":
    main()
