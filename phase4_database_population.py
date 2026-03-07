from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pybamm

from phase2_simulation import (
    determine_cells,
    load_cell_config,
    read_and_prepare_data,
    simulate_cells,
)


DEFAULT_DB_PATH = Path("outputs/phase4/battery_pipeline.db")
DEFAULT_SCHEMA_SQL_PATH = Path("phase3_database_schema.sql")
DEFAULT_CLEANED_CSV_PATH = Path("data/phase1_cleaned_data.csv")
DEFAULT_CELL_CONFIG_JSON_PATH = Path("data/phase2_cell_config.json")
DEFAULT_MODELS = ["SPM", "SPMe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate Phase 3 SQLite schema with cleaned experimental data and/or "
            "simulation data."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite DB path.",
    )
    parser.add_argument(
        "--schema-sql-path",
        type=Path,
        default=DEFAULT_SCHEMA_SQL_PATH,
        help="Path to SQL schema file used when creating/recreating the DB.",
    )
    parser.add_argument(
        "--cleaned-csv-path",
        type=Path,
        default=DEFAULT_CLEANED_CSV_PATH,
        help="Cleaned data CSV from Phase 1.",
    )
    parser.add_argument(
        "--cell-config-json",
        type=Path,
        default=DEFAULT_CELL_CONFIG_JSON_PATH,
        help="Cell config JSON with capacity_ah and initial_soc per cell.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "experimental-only", "simulation-only"],
        default="full",
        help=(
            "full: populate experiment + simulation; "
            "experimental-only: only cleaned data; "
            "simulation-only: only simulation data (for incremental appends)."
        ),
    )
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        help="Delete and recreate DB from schema before population.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Optional subset of cells to populate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["SPM", "SPMe", "DFN"],
        default=DEFAULT_MODELS,
        help="Models to run for simulation population.",
    )
    parser.add_argument(
        "--parameter-set",
        default="Chen2020",
        help="Base PyBaMM parameter set name (e.g., Chen2020, Chen2020_composite).",
    )
    parser.add_argument(
        "--parameter-name-extention",
        default="",
        help=(
            "Parameter set suffix stored in DB for tracking variants "
            '(e.g., "_tuned_v1").'
        ),
    )
    parser.add_argument(
        "--modified-parameters-json",
        type=Path,
        default=None,
        help="Optional JSON file persisted to parameter_sets.modified_parameters_json.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional simulation run label. Defaults to generated UTC timestamp label.",
    )
    parser.add_argument(
        "--replace-existing-experimental",
        action="store_true",
        help="Delete existing experimental runs/timeseries for selected cells before insert.",
    )
    parser.add_argument(
        "--replace-existing-simulation",
        action="store_true",
        help=(
            "Delete existing simulation run(s) for selected cells matching "
            "(model, parameter_set, name_extention, run_name) before insert."
        ),
    )
    parser.add_argument(
        "--max-cycle",
        type=float,
        default=None,
        help="Optional inclusive max cycle index for simulation replay.",
    )
    parser.add_argument(
        "--solver-mode",
        choices=["safe", "fast"],
        default="safe",
        help="PyBaMM CasadiSolver mode for simulation.",
    )
    parser.add_argument(
        "--voltage-max",
        type=float,
        default=4.5,
        help="Upper voltage cut-off [V] for simulation.",
    )
    parser.add_argument(
        "--voltage-min",
        type=float,
        default=2.0,
        help="Lower voltage cut-off [V] for simulation.",
    )
    parser.add_argument(
        "--profile",
        default="cleaned lab cycling profile",
        help="experimental_runs.profile value.",
    )
    parser.add_argument(
        "--environment",
        default="unknown",
        help="experimental_runs.environment value.",
    )
    return parser.parse_args()


def utc_now_run_name() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_schema(conn: sqlite3.Connection, schema_sql_path: Path) -> None:
    if not schema_sql_path.exists():
        raise FileNotFoundError(f"Schema SQL not found: {schema_sql_path}")
    schema_sql = schema_sql_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.commit()


def maybe_recreate_db(db_path: Path, schema_sql_path: Path, recreate_db: bool) -> None:
    if recreate_db and db_path.exists():
        db_path.unlink()

    must_create = not db_path.exists()
    conn = connect_db(db_path)
    try:
        if must_create or recreate_db:
            create_schema(conn, schema_sql_path)
        else:
            # Verify required tables exist.
            required = {
                "cells",
                "experimental_runs",
                "experimental_timeseries",
                "parameter_sets",
                "simulation_runs",
                "simulation_timeseries",
            }
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
            existing = {row[0] for row in rows}
            missing = sorted(required - existing)
            if missing:
                raise RuntimeError(
                    "DB exists but required tables are missing. "
                    f"Missing tables: {missing}. Use --recreate-db."
                )
    finally:
        conn.close()


def get_or_create_cell_id(conn: sqlite3.Connection, cell_code: str) -> int:
    row = conn.execute("SELECT id FROM cells WHERE cell_code = ?;", (cell_code,)).fetchone()
    if row is not None:
        return int(row[0])
    cur = conn.execute("INSERT INTO cells (cell_code) VALUES (?);", (cell_code,))
    return int(cur.lastrowid)


def load_modified_parameters_json(path: Path | None) -> str:
    if path is None:
        return "{}"
    if not path.exists():
        raise FileNotFoundError(f"modified-parameters JSON not found: {path}")
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    return json.dumps(parsed, separators=(",", ":"), sort_keys=True)


def parse_parameter_overrides(modified_parameters_json: str) -> dict[str, object]:
    parsed = json.loads(modified_parameters_json)
    if not isinstance(parsed, dict):
        raise ValueError("modified_parameters_json must be a JSON object.")
    out: dict[str, object] = {}
    for key, value in parsed.items():
        out[str(key)] = value
    return out


def ensure_simulation_run_parameter_overrides_column(conn: sqlite3.Connection) -> None:
    table_info = conn.execute("PRAGMA table_info(simulation_runs);").fetchall()
    column_names = {str(row[1]) for row in table_info}
    if "parameter_overrides_json" in column_names:
        return
    conn.execute(
        """
        ALTER TABLE simulation_runs
        ADD COLUMN parameter_overrides_json TEXT NOT NULL DEFAULT '{}';
        """
    )


def _jsonify_param_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify_param_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify_param_value(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def build_parameter_snapshots(
    base_parameter_set_name: str,
    modified_parameters_json: str,
) -> tuple[str, str]:
    overrides = parse_parameter_overrides(modified_parameters_json)
    base_values = pybamm.ParameterValues(base_parameter_set_name)

    base_dict = {
        str(k): _jsonify_param_value(v)
        for k, v in dict(base_values.items()).items()
    }
    overrides_json = json.dumps(overrides, separators=(",", ":"), sort_keys=True)
    base_json = json.dumps(base_dict, separators=(",", ":"), sort_keys=True)
    return base_json, overrides_json


def ensure_parameter_set_storage_columns(conn: sqlite3.Connection) -> int:
    table_info = conn.execute("PRAGMA table_info(parameter_sets);").fetchall()
    column_names = {str(row[1]) for row in table_info}
    if "base_parameters_json" not in column_names:
        conn.execute(
            """
            ALTER TABLE parameter_sets
            ADD COLUMN base_parameters_json TEXT NOT NULL DEFAULT '{}';
            """
        )

    rows = conn.execute(
        """
        SELECT
            id,
            base_parameter_set_name,
            modified_parameters_json
        FROM parameter_sets;
        """
    ).fetchall()
    for row in rows:
        parameter_set_id = int(row[0])
        base_name = str(row[1])
        modified_json = str(row[2] or "{}")
        base_json, normalized_overrides_json = build_parameter_snapshots(
            base_parameter_set_name=base_name,
            modified_parameters_json=modified_json,
        )
        conn.execute(
            """
            UPDATE parameter_sets
            SET modified_parameters_json = ?,
                base_parameters_json = ?
            WHERE id = ?;
            """,
            (
                normalized_overrides_json,
                base_json,
                parameter_set_id,
            ),
        )
    return len(rows)


def get_or_create_parameter_set_id(
    conn: sqlite3.Connection,
    base_parameter_set_name: str,
    name_extention: str,
    modified_parameters_json: str,
) -> int:
    base_json, overrides_json = build_parameter_snapshots(
        base_parameter_set_name=base_parameter_set_name,
        modified_parameters_json=modified_parameters_json,
    )
    row = conn.execute(
        """
        SELECT id
        FROM parameter_sets
        WHERE base_parameter_set_name = ? AND name_extention = ?;
        """,
        (base_parameter_set_name, name_extention),
    ).fetchone()
    if row is not None:
        parameter_set_id = int(row[0])
        conn.execute(
            """
            UPDATE parameter_sets
            SET modified_parameters_json = ?,
                base_parameters_json = ?
            WHERE id = ?;
            """,
            (overrides_json, base_json, parameter_set_id),
        )
        return parameter_set_id

    cur = conn.execute(
        """
        INSERT INTO parameter_sets (
            base_parameter_set_name,
            name_extention,
            modified_parameters_json,
            base_parameters_json
        )
        VALUES (?, ?, ?, ?);
        """,
        (
            base_parameter_set_name,
            name_extention,
            overrides_json,
            base_json,
        ),
    )
    return int(cur.lastrowid)


def populate_experimental_data(
    conn: sqlite3.Connection,
    cleaned_df: pd.DataFrame,
    selected_cells: list[str],
    profile: str,
    environment: str,
    replace_existing_experimental: bool,
) -> None:
    for cell_id in selected_cells:
        cell_df = cleaned_df[cleaned_df["cell_id"] == cell_id].copy()
        if cell_df.empty:
            raise ValueError(f"No cleaned rows found for cell_id={cell_id}")

        db_cell_id = get_or_create_cell_id(conn, cell_id)

        if replace_existing_experimental:
            conn.execute(
                "DELETE FROM experimental_runs WHERE cell_id = ?;",
                (db_cell_id,),
            )

        start_ts = pd.to_datetime(cell_df["datetime"]).min()
        end_ts = pd.to_datetime(cell_df["datetime"]).max()
        start_iso = start_ts.isoformat()
        end_iso = end_ts.isoformat()

        run_cur = conn.execute(
            """
            INSERT INTO experimental_runs (
                cell_id,
                start_time_ts_utc,
                end_time_ts_utc,
                profile,
                environment
            )
            VALUES (?, ?, ?, ?, ?);
            """,
            (db_cell_id, start_iso, end_iso, profile, environment),
        )
        experimental_run_id = int(run_cur.lastrowid)

        t0 = pd.to_datetime(cell_df["datetime"]).iloc[0]
        test_time_s = (pd.to_datetime(cell_df["datetime"]) - t0).dt.total_seconds()
        insert_rows = [
            (
                experimental_run_id,
                float(ts),
                int(cycle),
                int(step),
                float(current_a) if pd.notna(current_a) else None,
                float(voltage_v) if pd.notna(voltage_v) else None,
            )
            for ts, cycle, step, current_a, voltage_v in zip(
                test_time_s.to_numpy(),
                cell_df["cycle"].to_numpy(),
                cell_df["step"].to_numpy(),
                cell_df["current_A"].to_numpy(),
                cell_df["voltage_V"].to_numpy(),
            )
        ]
        conn.executemany(
            """
            INSERT INTO experimental_timeseries (
                experimental_run_id,
                test_time_s,
                cycle_index,
                step_index,
                current_a,
                voltage_v
            )
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            insert_rows,
        )
        print(
            f"Inserted experimental data for {cell_id}: "
            f"{len(insert_rows)} points (run_id={experimental_run_id})"
        )


def maybe_replace_existing_simulation_runs(
    conn: sqlite3.Connection,
    db_cell_id: int,
    parameter_set_id: int,
    model_name: str,
    run_name: str,
    replace_existing_simulation: bool,
) -> None:
    if not replace_existing_simulation:
        return
    conn.execute(
        """
        DELETE FROM simulation_runs
        WHERE cell_id = ?
          AND parameter_set_id = ?
          AND model_name = ?
          AND run_name = ?;
        """,
        (db_cell_id, parameter_set_id, model_name, run_name),
    )


def set_default_parameter_set_for_cells(
    conn: sqlite3.Connection,
    selected_cells: list[str],
    default_parameter_set_id: int,
) -> None:
    exists = conn.execute(
        "SELECT 1 FROM parameter_sets WHERE id = ?;",
        (default_parameter_set_id,),
    ).fetchone()
    if exists is None:
        raise RuntimeError(
            "Requested default_parameter_set_id does not exist in parameter_sets: "
            f"{default_parameter_set_id}"
        )

    for cell_id in selected_cells:
        db_cell_id = get_or_create_cell_id(conn, cell_id)
        conn.execute(
            """
            UPDATE cells
            SET default_parameter_set_id = ?
            WHERE id = ?;
            """,
            (default_parameter_set_id, db_cell_id),
        )


def populate_simulation_data(
    conn: sqlite3.Connection,
    cleaned_df: pd.DataFrame,
    selected_cells: list[str],
    config_map: dict[str, dict[str, float]],
    models: list[str],
    parameter_set: str,
    parameter_name_extention: str,
    modified_parameters_json: str,
    run_name: str,
    max_cycle: float | None,
    solver_mode: str,
    voltage_max: float | None,
    voltage_min: float | None,
    replace_existing_simulation: bool,
) -> None:
    parameter_overrides = parse_parameter_overrides(modified_parameters_json)
    parameter_set_id = get_or_create_parameter_set_id(
        conn=conn,
        base_parameter_set_name=parameter_set,
        name_extention=parameter_name_extention,
        modified_parameters_json=modified_parameters_json,
    )
    if parameter_name_extention == "":
        # Keep a single explicit "default" pointer on cells as requested.
        set_default_parameter_set_for_cells(
            conn=conn,
            selected_cells=selected_cells,
            default_parameter_set_id=1,
        )

    for model_name in models:
        print(f"Running simulation model={model_name} for cells={selected_cells}")
        summary_df, comparison_by_cell = simulate_cells(
            df=cleaned_df,
            cells=selected_cells,
            config_map=config_map,
            fallback_capacity_ah=None,
            fallback_initial_soc=None,
            output_dir=None,
            max_cycle=max_cycle,
            model_name=model_name,
            parameter_set=parameter_set,
            solver_mode=solver_mode,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            save_files=False,
            parameter_overrides=parameter_overrides,
        )
        summary_map = {
            str(row["cell_id"]): row for _, row in summary_df.iterrows()
        }

        for cell_id in selected_cells:
            if cell_id not in comparison_by_cell:
                raise RuntimeError(
                    f"Missing simulation comparison data for cell_id={cell_id}, model={model_name}"
                )
            db_cell_id = get_or_create_cell_id(conn, cell_id)
            maybe_replace_existing_simulation_runs(
                conn=conn,
                db_cell_id=db_cell_id,
                parameter_set_id=parameter_set_id,
                model_name=model_name,
                run_name=run_name,
                replace_existing_simulation=replace_existing_simulation,
            )

            row = summary_map[cell_id]
            sim_cur = conn.execute(
                """
                INSERT INTO simulation_runs (
                    cell_id,
                    parameter_set_id,
                    model_name,
                    capacity_ah,
                    initial_soc,
                    parameter_overrides_json,
                    run_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    db_cell_id,
                    parameter_set_id,
                    model_name,
                    float(row["capacity_ah"]),
                    float(row["initial_soc"]),
                    str(row.get("parameter_overrides_json", "{}")),
                    run_name,
                ),
            )
            simulation_run_id = int(sim_cur.lastrowid)
            comp = comparison_by_cell[cell_id]

            sim_rows = [
                (
                    simulation_run_id,
                    float(test_time_s),
                    int(cycle) if pd.notna(cycle) else None,
                    int(step) if pd.notna(step) else None,
                    float(current_a) if pd.notna(current_a) else None,
                    float(voltage_v) if pd.notna(voltage_v) else None,
                )
                for test_time_s, cycle, step, current_a, voltage_v in zip(
                    comp["time_s"].to_numpy(),
                    comp["cycle"].to_numpy(),
                    comp["step"].to_numpy(),
                    comp["current_A_sim_input"].to_numpy(),
                    comp["voltage_sim_V"].to_numpy(),
                )
            ]
            conn.executemany(
                """
                INSERT INTO simulation_timeseries (
                    simulation_run_id,
                    test_time_s,
                    cycle_index,
                    step_index,
                    current_a,
                    voltage_v
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                sim_rows,
            )
            print(
                f"Inserted simulation data for {cell_id} [{model_name}]: "
                f"{len(sim_rows)} points (run_id={simulation_run_id}, run_name={run_name})"
            )


def main() -> None:
    args = parse_args()
    run_name = args.run_name or utc_now_run_name()

    maybe_recreate_db(
        db_path=args.db_path,
        schema_sql_path=args.schema_sql_path,
        recreate_db=args.recreate_db,
    )

    cleaned_df = read_and_prepare_data(args.cleaned_csv_path)
    config_map = load_cell_config(args.cell_config_json)
    selected_cells = determine_cells(
        df=cleaned_df,
        requested_cells=args.cells,
        config_map=config_map,
    )
    modified_parameters_json = load_modified_parameters_json(args.modified_parameters_json)

    with connect_db(args.db_path) as conn:
        ensure_simulation_run_parameter_overrides_column(conn)
        ensure_parameter_set_storage_columns(conn)
        if args.mode in ("full", "experimental-only"):
            print("Populating cleaned experimental data...")
            populate_experimental_data(
                conn=conn,
                cleaned_df=cleaned_df,
                selected_cells=selected_cells,
                profile=args.profile,
                environment=args.environment,
                replace_existing_experimental=args.replace_existing_experimental,
            )

        if args.mode in ("full", "simulation-only"):
            print("Populating simulation data...")
            populate_simulation_data(
                conn=conn,
                cleaned_df=cleaned_df,
                selected_cells=selected_cells,
                config_map=config_map,
                models=args.models,
                parameter_set=args.parameter_set,
                parameter_name_extention=args.parameter_name_extention,
                modified_parameters_json=modified_parameters_json,
                run_name=run_name,
                max_cycle=args.max_cycle,
                solver_mode=args.solver_mode,
                voltage_max=args.voltage_max,
                voltage_min=args.voltage_min,
                replace_existing_simulation=args.replace_existing_simulation,
            )

        conn.commit()

    print(f"Done. DB populated at: {args.db_path}")
    print(f"Mode: {args.mode}")
    print(f"Cells: {selected_cells}")
    if args.mode in ("full", "simulation-only"):
        print(f"Simulation models: {args.models}")
        print(f"Simulation run_name: {run_name}")


if __name__ == "__main__":
    main()
