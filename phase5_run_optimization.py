from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase5_cached_runner import build_cached_runner, solve_with_inputs
from phase5_common import (
    DEFAULT_CELL_CONFIG_JSON_PATH,
    DIFFUSION_CANDIDATES,
    OHMIC_CANDIDATES,
    REACTION_CANDIDATES,
    SELECTOR_CHOICES,
    apply_scaled_override,
    build_rmse_weights,
    filter_cell_cycles,
    parameter_name_from_selector,
    parse_cycle_range,
    prepare_base_parameter_context,
    print_phase5_context,
    resolve_capacity_and_initial_soc,
    validate_base_inputs,
)
from phase5_optimization_core import (
    OptimizationVariable,
    evaluate_window_rmses,
    run_coarse_global_optimization,
    run_local_refinement,
    vector_to_parameter_dict,
    weighted_window_rmse,
)
from phase2_simulation import read_and_prepare_data


INPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
OUTPUT_DIR = Path("outputs/phase5")
DEFAULT_DB_PATH = Path("outputs/phase4/battery_pipeline.db")
DEFAULT_SCHEMA_SQL_PATH = Path("phase3_database_schema.sql")
DEFAULT_PHASE4_OUTPUT_DIR = Path("outputs/phase4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 optimization: fit selected parameters for one cell and one cycle range "
            "using category-specific RMSE windows."
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

    parser.add_argument("--optimize-initial-soc", action="store_true")
    parser.add_argument("--optimize-capacity", action="store_true")
    parser.add_argument(
        "--ohmic-parameter",
        type=str,
        choices=SELECTOR_CHOICES,
        default=None,
        help="Ohmic parameter location selector: n (negative), p (positive), e (electrolyte).",
    )
    parser.add_argument(
        "--kinetic-parameter",
        type=str,
        choices=SELECTOR_CHOICES,
        default=None,
        help="Kinetic parameter location selector: n or p (e is not supported).",
    )
    parser.add_argument(
        "--diffusion-parameter",
        type=str,
        choices=SELECTOR_CHOICES,
        default=None,
        help="Diffusion parameter location selector: n (negative), p (positive), e (electrolyte).",
    )
    parser.add_argument(
        "--initial-soc-scale-bounds",
        nargs=2,
        type=float,
        default=[0.8, 1.2],
        help="Bounds for (optimized initial_soc / input initial_soc).",
    )
    parser.add_argument(
        "--capacity-scale-bounds",
        nargs=2,
        type=float,
        default=[0.6, 1.6],
        help="Bounds for (optimized capacity / input capacity-ah).",
    )
    parser.add_argument("--ohmic-scale-bounds", nargs=2, type=float, default=[0.5, 2.0])
    parser.add_argument("--kinetic-scale-bounds", nargs=2, type=float, default=[0.2, 5.0])
    parser.add_argument("--diffusion-scale-bounds", nargs=2, type=float, default=[0.2, 5.0])
    parser.add_argument(
        "--rmse-weights",
        nargs=4,
        type=float,
        dest="rmse_weights",
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=("OHMIC", "KINETIC", "DIFFUSION", "CAPACITY"),
        help="Weights for window RMSEs in weighted-sum objective.",
    )

    parser.add_argument("--stage1-maxiter", type=int, default=8)
    parser.add_argument("--stage1-popsize", type=int, default=6)
    parser.add_argument("--stage1-seed", type=int, default=42)
    parser.add_argument(
        "--stage1-variable-limit",
        type=int,
        default=3,
        help="Deprecated: stage-1 now optimizes all selected variables.",
    )
    parser.add_argument(
        "--local-method",
        choices=["L-BFGS-B", "Powell"],
        default="Powell",
        help="Stage-2 local optimization method.",
    )
    parser.add_argument(
        "--stage2-maxiter",
        type=int,
        default=40,
        help="Maximum iterations for stage-2 local refinement.",
    )
    parser.add_argument(
        "--tail-downsample-stride-stage1",
        type=int,
        default=60,
        help=(
            "Stage-1 capacity-window RMSE downsampling stride (apply every Nth point only "
            "for 120s+ within each step; simulation still runs on full time grid)."
        ),
    )
    parser.add_argument(
        "--tail-downsample-stride-stage2",
        type=int,
        default=60,
        help=(
            "Stage-2 capacity-window RMSE downsampling stride (apply every Nth point only "
            "for 120s+ within each step; simulation still runs on full time grid)."
        ),
    )
    parser.add_argument(
        "--penalty-rmse",
        type=float,
        dest="penalty_rmse",
        default=10.0,
        help="Penalty RMSE [V] used when simulation fails.",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--schema-sql-path", type=Path, default=DEFAULT_SCHEMA_SQL_PATH)
    parser.add_argument(
        "--db-parameter-name-extention",
        type=str,
        default="_phase5_optimized",
        help="Suffix stored in parameter_sets.name_extention for appended optimized runs.",
    )
    parser.add_argument(
        "--db-run-name",
        type=str,
        default=None,
        help="Optional run_name for appended optimized run. Defaults to generated UTC label.",
    )
    parser.add_argument(
        "--db-replace-existing-simulation",
        action="store_true",
        help=(
            "Delete matching simulation runs before insert when appending "
            "(same cell/model/parameter_set/name_extention/run_name)."
        ),
    )
    parser.add_argument("--phase4-output-dir", type=Path, default=DEFAULT_PHASE4_OUTPUT_DIR)
    parser.add_argument(
        "--plot-cycle",
        type=str,
        default=None,
        help="Cycle filter passed to phase4_plot_from_db.py. Defaults to --cycle when omitted.",
    )
    parser.add_argument(
        "--base-simulation-run-id",
        type=int,
        default=None,
        help=(
            "Optional simulation_runs.id used as optimization baseline "
            "(loads run parameter_overrides_json and, when capacity/SoC are omitted, "
            "uses run capacity_ah and initial_soc)."
        ),
    )
    parser.add_argument(
        "--base-run-name",
        type=str,
        default=None,
        help=(
            "Optional run_name used as optimization baseline (latest matching run for "
            "target cell/model/parameter_set; optionally narrowed by "
            "--base-run-parameter-name-extention)."
        ),
    )
    parser.add_argument(
        "--base-run-parameter-name-extention",
        type=str,
        default=None,
        help="Optional name_extention filter used with --base-run-name.",
    )
    return parser.parse_args()


def build_variables(args: argparse.Namespace) -> list[OptimizationVariable]:
    variables: list[OptimizationVariable] = []
    if args.optimize_initial_soc:
        soc_scale_lo, soc_scale_hi = float(args.initial_soc_scale_bounds[0]), float(
            args.initial_soc_scale_bounds[1]
        )
        soc_lo = max(0.0, float(args.initial_soc) * soc_scale_lo)
        soc_hi = min(1.0, float(args.initial_soc) * soc_scale_hi)
        if soc_hi <= soc_lo:
            raise ValueError(
                "Invalid initial SoC scale bounds after clipping to [0,1]. "
                f"baseline={args.initial_soc}, scale_bounds={args.initial_soc_scale_bounds}, "
                f"resolved_bounds=({soc_lo}, {soc_hi})"
            )
        variables.append(
            OptimizationVariable(
                name="initial_soc",
                kind="initial_soc",
                lower=soc_lo,
                upper=soc_hi,
                initial=float(args.initial_soc),
            )
        )
    if args.optimize_capacity:
        scale_lo, scale_hi = float(args.capacity_scale_bounds[0]), float(
            args.capacity_scale_bounds[1]
        )
        variables.append(
            OptimizationVariable(
                name="capacity_scale",
                kind="capacity_scale",
                lower=scale_lo,
                upper=scale_hi,
                initial=1.0,
            )
        )
    if args.ohmic_parameter is not None:
        ohmic_name = parameter_name_from_selector("ohmic", str(args.ohmic_parameter))
        variables.append(
            OptimizationVariable(
                name="ohmic_scale",
                kind="parameter_scale",
                lower=float(args.ohmic_scale_bounds[0]),
                upper=float(args.ohmic_scale_bounds[1]),
                initial=1.0,
                parameter_name=ohmic_name,
            )
        )
    if args.kinetic_parameter is not None:
        kinetic_name = parameter_name_from_selector("kinetic", str(args.kinetic_parameter))
        variables.append(
            OptimizationVariable(
                name="kinetic_scale",
                kind="parameter_scale",
                lower=float(args.kinetic_scale_bounds[0]),
                upper=float(args.kinetic_scale_bounds[1]),
                initial=1.0,
                parameter_name=kinetic_name,
            )
        )
    if args.diffusion_parameter is not None:
        diffusion_name = parameter_name_from_selector("diffusion", str(args.diffusion_parameter))
        variables.append(
            OptimizationVariable(
                name="diffusion_scale",
                kind="parameter_scale",
                lower=float(args.diffusion_scale_bounds[0]),
                upper=float(args.diffusion_scale_bounds[1]),
                initial=1.0,
                parameter_name=diffusion_name,
            )
        )
    return variables


def _format_best_summary(best: dict[str, object], variables: list[OptimizationVariable]) -> str:
    objective_key = "objective_stage2_fit_V" if "objective_stage2_fit_V" in best else "objective_V"
    if objective_key not in best or not np.isfinite(float(best[objective_key])):
        return "best objective not available yet"
    parts = [
        f"objective={float(best[objective_key]):.6f} V",
        f"capacity_ah={float(best['capacity_ah']):.6f}",
        f"initial_soc={float(best['initial_soc']):.6f}",
    ]
    for var in variables:
        value = best.get(f"var_{var.name}", None)
        if value is None:
            continue
        label = f"var_{var.name}"
        if var.parameter_name is None:
            parts.append(f"{label}={float(value):.6f}")
        else:
            parts.append(f"{label}={float(value):.6f} ({var.parameter_name})")
    return ", ".join(parts)


def print_optimization_configuration(
    args: argparse.Namespace,
    variables: list[OptimizationVariable],
) -> None:
    print("Optimization run configuration:")
    print(f"  baseline capacity_ah: {float(args.capacity_ah):.6f}")
    print(f"  baseline initial_soc: {float(args.initial_soc):.6f}")
    print("  selected variables:")
    for var in variables:
        parameter_note = (
            f", parameter={var.parameter_name}" if var.parameter_name is not None else ""
        )
        print(
            f"    - {var.name} [{var.kind}] "
            f"bounds=({var.lower:.6g}, {var.upper:.6g}), initial={var.initial:.6g}"
            f"{parameter_note}"
        )
    print(
        "  stage-1 global settings: "
        f"maxiter={int(args.stage1_maxiter)}, popsize={int(args.stage1_popsize)}, "
        f"seed={int(args.stage1_seed)}"
    )


def utc_now_run_name() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _is_numeric_scalar(value: object) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if hasattr(value, "item"):
        try:
            _ = value.item()
            return True
        except Exception:
            return False
    return False


def _scaled_parameter_value(base_value: object, scale: float) -> object:
    if callable(base_value):
        def wrapped(*args: object, _base=base_value, _scale=float(scale)) -> object:
            return _scale * _base(*args)

        return wrapped
    if _is_numeric_scalar(base_value):
        scalar = base_value.item() if hasattr(base_value, "item") else base_value
        return float(scale) * float(scalar)
    raise ValueError(
        "Unsupported parameter type for scaling. "
        f"type={type(base_value).__name__}"
    )


def _normalize_parameter_overrides(raw_json: str) -> dict[str, object]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("parameter_overrides_json must be an object.")
    out: dict[str, object] = {}
    for key, value in parsed.items():
        out[str(key)] = value
    return out


def resolve_base_run_overrides(args: argparse.Namespace) -> tuple[dict[str, object], int | None]:
    if args.base_simulation_run_id is None and args.base_run_name is None:
        return {}, None
    if args.base_simulation_run_id is not None and args.base_run_name is not None:
        raise ValueError("Use either --base-simulation-run-id or --base-run-name, not both.")
    with sqlite3.connect(args.db_path) as conn:
        if args.base_simulation_run_id is not None:
            row = conn.execute(
                """
                SELECT
                    sr.id,
                    c.cell_code,
                    sr.model_name,
                    ps.base_parameter_set_name,
                    ps.name_extention,
                    sr.run_name,
                    sr.capacity_ah,
                    sr.initial_soc,
                    COALESCE(sr.parameter_overrides_json, '{}')
                FROM simulation_runs sr
                JOIN cells c ON c.id = sr.cell_id
                JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
                WHERE sr.id = ?
                LIMIT 1;
                """,
                (int(args.base_simulation_run_id),),
            ).fetchone()
        else:
            params: list[object] = [str(args.cell), str(args.model_name), str(args.parameter_set), str(args.base_run_name)]
            ext_clause = ""
            if args.base_run_parameter_name_extention is not None:
                ext_clause = " AND ps.name_extention = ?"
                params.append(str(args.base_run_parameter_name_extention))
            row = conn.execute(
                f"""
                SELECT
                    sr.id,
                    c.cell_code,
                    sr.model_name,
                    ps.base_parameter_set_name,
                    ps.name_extention,
                    sr.run_name,
                    sr.capacity_ah,
                    sr.initial_soc,
                    COALESCE(sr.parameter_overrides_json, '{{}}')
                FROM simulation_runs sr
                JOIN cells c ON c.id = sr.cell_id
                JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
                WHERE c.cell_code = ?
                  AND sr.model_name = ?
                  AND ps.base_parameter_set_name = ?
                  AND sr.run_name = ?
                  {ext_clause}
                ORDER BY sr.id DESC
                LIMIT 1;
                """,
                params,
            ).fetchone()
    if row is None:
        raise RuntimeError("Requested base simulation run was not found in DB.")

    base_run_id = int(row[0])
    base_cell = str(row[1])
    base_model = str(row[2])
    base_parameter_set = str(row[3])
    base_name_extention = str(row[4] or "")
    base_run_name = str(row[5] or "")
    base_capacity = float(row[6]) if row[6] is not None else None
    base_soc = float(row[7]) if row[7] is not None else None
    raw_overrides = _normalize_parameter_overrides(str(row[8] or "{}"))

    if base_cell != str(args.cell):
        raise ValueError(
            f"Base run cell mismatch: requested cell={args.cell}, base run cell={base_cell}."
        )
    if base_model != str(args.model_name):
        raise ValueError(
            f"Base run model mismatch: requested model={args.model_name}, base run model={base_model}."
        )
    if base_parameter_set != str(args.parameter_set):
        raise ValueError(
            "Base run parameter-set mismatch: "
            f"requested={args.parameter_set}, base run={base_parameter_set}."
        )

    print(
        "Using base run from DB: "
        f"id={base_run_id}, run_name={base_run_name}, name_extention={base_name_extention}"
    )
    if args.capacity_ah is None and base_capacity is not None:
        args.capacity_ah = float(base_capacity)
    if args.initial_soc is None and base_soc is not None:
        args.initial_soc = float(base_soc)

    # Capacity scaling is controlled by capacity_ah and should not be copied as fixed override.
    raw_overrides.pop("Nominal cell capacity [A.h]", None)
    raw_overrides.pop("Number of electrodes connected in parallel to make a cell", None)
    return raw_overrides, base_run_id


def prepare_parameter_overrides_for_optimization(
    base_overrides: dict[str, object],
    candidate_base_values: dict[str, object],
    optimized_parameter_names: set[str],
) -> tuple[dict[str, object], dict[str, object], dict[str, float]]:
    adjusted_base_values = dict(candidate_base_values)
    callable_base_scales: dict[str, float] = {}
    fixed_parameter_overrides: dict[str, object] = {}
    for parameter_name, override in base_overrides.items():
        if parameter_name in candidate_base_values:
            base_value = candidate_base_values[parameter_name]
            if isinstance(override, dict) and override.get("mode") == "scale":
                scale = float(override.get("value", 1.0))
                adjusted_base_values[parameter_name] = _scaled_parameter_value(base_value, scale)
                if callable(base_value):
                    callable_base_scales[parameter_name] = scale
            else:
                adjusted_base_values[parameter_name] = float(override)
        if parameter_name not in optimized_parameter_names:
            fixed_parameter_overrides[parameter_name] = override
    return adjusted_base_values, fixed_parameter_overrides, callable_base_scales


def build_db_modified_parameters(
    best_result: dict[str, object],
    fixed_parameter_overrides: dict[str, object],
    original_base_value_map: dict[str, object],
    adjusted_base_value_map: dict[str, object],
    callable_base_scales: dict[str, float],
) -> dict[str, object]:
    parameter_scales = best_result.get("parameter_scales", {})
    if not isinstance(parameter_scales, dict):
        raise ValueError("best_result.parameter_scales is missing or invalid.")
    modified: dict[str, object] = dict(fixed_parameter_overrides)
    for parameter_name, scale_obj in parameter_scales.items():
        scale = float(scale_obj)
        if parameter_name not in original_base_value_map or parameter_name not in adjusted_base_value_map:
            raise KeyError(f"Missing base parameter value for '{parameter_name}'")
        original_base = original_base_value_map[parameter_name]
        adjusted_base = adjusted_base_value_map[parameter_name]
        if callable(original_base):
            prior_scale = float(callable_base_scales.get(parameter_name, 1.0))
            modified[str(parameter_name)] = {"mode": "scale", "value": prior_scale * scale}
            continue
        if _is_numeric_scalar(adjusted_base):
            scalar = adjusted_base.item() if hasattr(adjusted_base, "item") else adjusted_base
            modified[str(parameter_name)] = float(scalar) * scale
            continue
        raise ValueError(
            f"Unsupported base parameter type for DB export: {parameter_name} "
            f"({type(adjusted_base).__name__})"
        )
    return modified


def write_db_artifacts(
    output_dir: Path,
    cell_id: str,
    best_result: dict[str, object],
    fixed_parameter_overrides: dict[str, object],
    original_base_value_map: dict[str, object],
    adjusted_base_value_map: dict[str, object],
    callable_base_scales: dict[str, float],
) -> tuple[Path, Path]:
    modified_parameters = build_db_modified_parameters(
        best_result=best_result,
        fixed_parameter_overrides=fixed_parameter_overrides,
        original_base_value_map=original_base_value_map,
        adjusted_base_value_map=adjusted_base_value_map,
        callable_base_scales=callable_base_scales,
    )
    modified_path = output_dir / f"{cell_id}_db_modified_parameters.json"
    with modified_path.open("w", encoding="utf-8") as f:
        json.dump(modified_parameters, f, indent=2)
        f.write("\n")

    optimized_cell_config = {
        str(cell_id): {
            "capacity_ah": float(best_result["capacity_ah"]),
            "initial_soc": float(best_result["initial_soc"]),
        }
    }
    config_path = output_dir / f"{cell_id}_optimized_cell_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(optimized_cell_config, f, indent=2)
        f.write("\n")
    return modified_path, config_path


def run_phase4_append_to_db(
    args: argparse.Namespace,
    modified_parameters_json_path: Path,
    optimized_cell_config_path: Path,
    parameter_name_extention: str,
    run_name: str,
) -> None:
    cmd = [
        sys.executable,
        "phase4_database_population.py",
        "--db-path",
        str(args.db_path),
        "--schema-sql-path",
        str(args.schema_sql_path),
        "--cleaned-csv-path",
        str(args.input_csv),
        "--cell-config-json",
        str(optimized_cell_config_path),
        "--mode",
        "simulation-only",
        "--cells",
        str(args.cell),
        "--models",
        str(args.model_name),
        "--parameter-set",
        str(args.parameter_set),
        "--parameter-name-extention",
        str(parameter_name_extention),
        "--modified-parameters-json",
        str(modified_parameters_json_path),
        "--run-name",
        str(run_name),
        "--solver-mode",
        str(args.solver_mode),
        "--voltage-max",
        str(args.voltage_max),
        "--voltage-min",
        str(args.voltage_min),
    ]
    if args.cycle is not None:
        cycle_range = parse_cycle_range(args.cycle)
        if cycle_range is not None:
            cmd.extend(["--max-cycle", str(cycle_range[1])])
    if args.db_replace_existing_simulation:
        cmd.append("--replace-existing-simulation")
    subprocess.run(cmd, check=True)


def resolve_safe_parameter_name_extention(
    db_path: Path,
    base_parameter_set_name: str,
    requested_name_extention: str,
    modified_parameters_json_path: Path,
) -> str:
    requested = str(requested_name_extention)
    raw = modified_parameters_json_path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    normalized_new = json.dumps(parsed, separators=(",", ":"), sort_keys=True)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT modified_parameters_json
            FROM parameter_sets
            WHERE base_parameter_set_name = ? AND name_extention = ?
            LIMIT 1;
            """,
            (base_parameter_set_name, requested),
        ).fetchone()
    if row is None:
        return requested
    existing = json.loads(str(row[0] or "{}"))
    normalized_existing = json.dumps(existing, separators=(",", ":"), sort_keys=True)
    if normalized_existing == normalized_new:
        return requested
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved = f"{requested}__{suffix}"
    print(
        "Detected existing parameter-set name_extention with different overrides; "
        f"using collision-safe name_extention='{resolved}'."
    )
    return resolved


def resolve_optimized_run_id(
    db_path: Path,
    cell_id: str,
    model_name: str,
    parameter_set: str,
    optimized_run_name: str,
) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT sr.id
            FROM simulation_runs sr
            JOIN cells c ON c.id = sr.cell_id
            JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
            WHERE c.cell_code = ?
              AND sr.model_name = ?
              AND ps.base_parameter_set_name = ?
              AND sr.run_name = ?
            ORDER BY sr.id DESC
            LIMIT 1;
            """,
            (cell_id, model_name, parameter_set, optimized_run_name),
        ).fetchone()
    if row is None:
        raise RuntimeError("Optimized run was not found after DB append.")
    return int(row[0])


def resolve_comparison_run_ids(
    db_path: Path,
    cell_id: str,
    model_name: str,
    parameter_set: str,
    optimized_run_id: int,
) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        baseline_row = conn.execute(
            """
            SELECT sr.id
            FROM simulation_runs sr
            JOIN cells c ON c.id = sr.cell_id
            JOIN parameter_sets ps ON ps.id = sr.parameter_set_id
            WHERE c.cell_code = ?
              AND sr.model_name = ?
              AND ps.base_parameter_set_name = ?
              AND ps.name_extention = ''
            ORDER BY sr.id DESC
            LIMIT 1;
            """,
            (cell_id, model_name, parameter_set),
        ).fetchone()

    if baseline_row is None:
        return [optimized_run_id]
    baseline_run_id = int(baseline_row[0])
    if baseline_run_id == optimized_run_id:
        return [optimized_run_id]
    return [baseline_run_id, optimized_run_id]


def ensure_optimization_tracking_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY,
            simulation_run_id INTEGER NOT NULL UNIQUE,
            base_simulation_run_id INTEGER,
            created_at_ts_utc TEXT NOT NULL,
            objective_v REAL,
            optimization_config_json TEXT NOT NULL DEFAULT '{}',
            best_result_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (simulation_run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE,
            FOREIGN KEY (base_simulation_run_id) REFERENCES simulation_runs(id) ON DELETE SET NULL
        );
        """
    )


def upsert_optimization_metadata(
    db_path: Path,
    simulation_run_id: int,
    base_simulation_run_id: int | None,
    optimization_config: dict[str, object],
    best_result: dict[str, object],
) -> None:
    with sqlite3.connect(db_path) as conn:
        ensure_optimization_tracking_table(conn)
        conn.execute(
            """
            INSERT INTO optimization_runs (
                simulation_run_id,
                base_simulation_run_id,
                created_at_ts_utc,
                objective_v,
                optimization_config_json,
                best_result_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(simulation_run_id) DO UPDATE SET
                base_simulation_run_id = excluded.base_simulation_run_id,
                created_at_ts_utc = excluded.created_at_ts_utc,
                objective_v = excluded.objective_v,
                optimization_config_json = excluded.optimization_config_json,
                best_result_json = excluded.best_result_json;
            """,
            (
                int(simulation_run_id),
                None if base_simulation_run_id is None else int(base_simulation_run_id),
                datetime.now(timezone.utc).isoformat(),
                float(
                    best_result.get(
                        "objective_stage2_fit_V",
                        best_result.get("objective_V", float("nan")),
                    )
                ),
                json.dumps(optimization_config, separators=(",", ":"), sort_keys=True),
                json.dumps(best_result, separators=(",", ":"), sort_keys=True),
            ),
        )
        conn.commit()


def run_phase4_plot_from_db(
    args: argparse.Namespace,
    simulation_run_ids: list[int],
) -> None:
    cmd = [
        sys.executable,
        "phase4_plot_from_db.py",
        "--db-path",
        str(args.db_path),
        "--output-dir",
        str(args.phase4_output_dir),
        "--simulation-run-ids",
        *[str(run_id) for run_id in simulation_run_ids],
    ]
    cycle_value = args.plot_cycle if args.plot_cycle is not None else args.cycle
    if cycle_value is not None:
        cmd.extend(["--cycle", str(cycle_value)])
    cmd.append("--plot-with-current")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cycle_range = parse_cycle_range(args.cycle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_run_overrides, base_simulation_run_id = resolve_base_run_overrides(args)

    capacity_ah, initial_soc = resolve_capacity_and_initial_soc(
        cell_config_json=args.cell_config_json,
        cell_id=args.cell,
        capacity_ah=args.capacity_ah,
        initial_soc=args.initial_soc,
    )
    args.capacity_ah = float(capacity_ah)
    args.initial_soc = float(initial_soc)
    validate_base_inputs(capacity_ah=float(args.capacity_ah), initial_soc=float(args.initial_soc))
    rmse_weights = build_rmse_weights(list(args.rmse_weights))
    bounds_to_check = {
        "initial-soc-scale-bounds": args.initial_soc_scale_bounds,
        "capacity-scale-bounds": args.capacity_scale_bounds,
        "ohmic-scale-bounds": args.ohmic_scale_bounds,
        "kinetic-scale-bounds": args.kinetic_scale_bounds,
        "diffusion-scale-bounds": args.diffusion_scale_bounds,
    }
    for name, bounds in bounds_to_check.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo <= 0 or hi <= 0:
            raise ValueError(f"--{name} must be positive, got {bounds}.")
        if hi <= lo:
            raise ValueError(f"--{name} must be ascending, got {bounds}.")

    if int(args.stage1_variable_limit) <= 0:
        raise ValueError("--stage1-variable-limit must be >= 1.")
    if int(args.tail_downsample_stride_stage1) <= 0:
        raise ValueError("--tail-downsample-stride-stage1 must be >= 1.")
    if int(args.tail_downsample_stride_stage2) <= 0:
        raise ValueError("--tail-downsample-stride-stage2 must be >= 1.")

    df = read_and_prepare_data(args.input_csv)
    cell_df_full = filter_cell_cycles(df=df, cell_id=args.cell, cycle_range=cycle_range)
    cell_df_stage1 = cell_df_full
    cell_df_stage2 = cell_df_full
    prepared_names = set(DIFFUSION_CANDIDATES + REACTION_CANDIDATES + OHMIC_CANDIDATES)
    base_value_map, missing_names = prepare_base_parameter_context(
        parameter_set=args.parameter_set,
        candidate_names=prepared_names,
    )
    print_phase5_context(
        target_label="Phase5 optimization target",
        cell_id=args.cell,
        cycle_range=cycle_range,
        model_name=args.model_name,
        parameter_set=args.parameter_set,
        n_points=len(cell_df_full),
        rmse_weights=rmse_weights,
        base_value_map=base_value_map,
        missing_names=missing_names,
    )
    print(
        "Evaluation data points (PyBaMM full-grid solve): "
        f"full={len(cell_df_full)}, stage1={len(cell_df_stage1)}, stage2={len(cell_df_stage2)}"
    )
    print(
        "Capacity-window RMSE tail downsample strides (120s+ per step): "
        f"stage1={int(args.tail_downsample_stride_stage1)}, "
        f"stage2={int(args.tail_downsample_stride_stage2)}"
    )

    selected_count = int(args.optimize_initial_soc) + int(args.optimize_capacity)
    selected_count += int(args.ohmic_parameter is not None)
    selected_count += int(args.kinetic_parameter is not None)
    selected_count += int(args.diffusion_parameter is not None)
    if selected_count == 0:
        raise ValueError(
            "No optimization variables selected. "
            "Use one or more of: --optimize-initial-soc, --optimize-capacity, "
            "--ohmic-parameter {n,p,e}, --kinetic-parameter {n,p}, "
            "--diffusion-parameter {n,p,e}."
        )

    variables = build_variables(args)
    optimized_parameter_names = {
        str(var.parameter_name)
        for var in variables
        if var.kind == "parameter_scale" and var.parameter_name is not None
    }
    adjusted_base_value_map, fixed_parameter_overrides, callable_base_scales = (
        prepare_parameter_overrides_for_optimization(
            base_overrides=base_run_overrides,
            candidate_base_values=base_value_map,
            optimized_parameter_names=optimized_parameter_names,
        )
    )
    if fixed_parameter_overrides:
        print(
            "Applying fixed base parameter overrides from selected base run "
            f"({len(fixed_parameter_overrides)} keys)."
        )
    print_optimization_configuration(args=args, variables=variables)

    for var in variables:
        if var.kind == "parameter_scale" and var.parameter_name not in adjusted_base_value_map:
            raise ValueError(
                f"Selected parameter is not available in {args.parameter_set}: {var.parameter_name}"
            )

    optimized_parameter_names = sorted(
        [
            str(var.parameter_name)
            for var in variables
            if var.kind == "parameter_scale" and var.parameter_name is not None
        ]
    )
    runner_stage1 = build_cached_runner(
        model_name=str(args.model_name),
        parameter_set=str(args.parameter_set),
        solver_mode=str(args.solver_mode),
        voltage_max=None if args.voltage_max is None else float(args.voltage_max),
        voltage_min=None if args.voltage_min is None else float(args.voltage_min),
        base_value_map=adjusted_base_value_map,
        current_profile_df=cell_df_full,
        eval_df=cell_df_stage1,
        fixed_parameter_overrides=fixed_parameter_overrides,
        optimized_parameter_names=optimized_parameter_names,
    )
    runner_stage2 = build_cached_runner(
        model_name=str(args.model_name),
        parameter_set=str(args.parameter_set),
        solver_mode=str(args.solver_mode),
        voltage_max=None if args.voltage_max is None else float(args.voltage_max),
        voltage_min=None if args.voltage_min is None else float(args.voltage_min),
        base_value_map=adjusted_base_value_map,
        current_profile_df=cell_df_full,
        eval_df=cell_df_stage2,
        fixed_parameter_overrides=fixed_parameter_overrides,
        optimized_parameter_names=optimized_parameter_names,
    )

    stage1_variables = list(variables)
    if int(args.stage1_variable_limit) != len(variables):
        print(
            "Info: --stage1-variable-limit is deprecated and ignored; "
            "stage-1 optimizes all selected variables."
        )

    stage1_best, stage1_history_df, stage1_result = run_coarse_global_optimization(
        runner=runner_stage1,
        variables=stage1_variables,
        baseline_capacity_ah=float(args.capacity_ah),
        baseline_initial_soc=float(args.initial_soc),
        rmse_weights=rmse_weights,
        penalty_rmse=float(args.penalty_rmse),
        capacity_tail_stride=int(args.tail_downsample_stride_stage1),
        stage1_maxiter=int(args.stage1_maxiter),
        stage1_popsize=int(args.stage1_popsize),
        stage1_seed=int(args.stage1_seed),
    )

    x0_stage2_by_name = {var.name: float(var.initial) for var in variables}
    for var in stage1_variables:
        key = f"var_{var.name}"
        if key in stage1_best and np.isfinite(float(stage1_best[key])):
            x0_stage2_by_name[var.name] = float(stage1_best[key])
    x0_stage2 = np.asarray([x0_stage2_by_name[var.name] for var in variables], dtype=float)

    stage2_best, stage2_history_df, stage2_result = run_local_refinement(
        runner=runner_stage2,
        variables=variables,
        baseline_capacity_ah=float(args.capacity_ah),
        baseline_initial_soc=float(args.initial_soc),
        rmse_weights=rmse_weights,
        penalty_rmse=float(args.penalty_rmse),
        capacity_tail_stride=int(args.tail_downsample_stride_stage2),
        x0=x0_stage2,
        local_method=str(args.local_method),
        stage2_maxiter=int(args.stage2_maxiter),
    )

    best_x = np.asarray(
        [
            float(stage2_best.get(f"var_{var.name}", x0_stage2_by_name[var.name]))
            for var in variables
        ],
        dtype=float,
    )
    best_mapping = vector_to_parameter_dict(
        x=best_x,
        variables=variables,
        baseline_capacity_ah=float(args.capacity_ah),
        baseline_initial_soc=float(args.initial_soc),
    )
    best_parameter_scales = dict(best_mapping["parameter_scales"])
    best_inputs = {
        "capacity_ah": float(best_mapping["capacity_ah"]),
        "initial_soc": float(best_mapping["initial_soc"]),
    }
    for parameter_name, scale in best_parameter_scales.items():
        best_inputs[f"scale::{parameter_name}"] = float(scale)

    runner_full = build_cached_runner(
        model_name=str(args.model_name),
        parameter_set=str(args.parameter_set),
        solver_mode=str(args.solver_mode),
        voltage_max=None if args.voltage_max is None else float(args.voltage_max),
        voltage_min=None if args.voltage_min is None else float(args.voltage_min),
        base_value_map=adjusted_base_value_map,
        current_profile_df=cell_df_full,
        eval_df=cell_df_full,
        fixed_parameter_overrides=fixed_parameter_overrides,
        optimized_parameter_names=optimized_parameter_names,
    )
    full_trace = solve_with_inputs(runner=runner_full, inputs=best_inputs)
    final_metrics = evaluate_window_rmses(
        full_trace,
        capacity_tail_stride=int(args.tail_downsample_stride_stage2),
    )
    objective_full = weighted_window_rmse(metrics=final_metrics, weights=rmse_weights)
    initial_soc_fallback_used = bool(
        runner_stage1.initial_soc_fallback_used
        or runner_stage2.initial_soc_fallback_used
        or runner_full.initial_soc_fallback_used
    )

    best_result = {
        **{k: v for k, v in stage2_best.items() if k not in {"stage", "objective_V"}},
        **final_metrics,
        "objective_stage2_fit_V": float(stage2_best["objective_V"]),
        "objective_full_profile_weighted_window_V": float(objective_full),
        "capacity_ah": float(best_mapping["capacity_ah"]),
        "initial_soc": float(best_mapping["initial_soc"]),
        "parameter_scales": best_parameter_scales,
        "fixed_parameter_overrides": dict(fixed_parameter_overrides),
        "stage1_optimizer_success": bool(getattr(stage1_result, "success", False)),
        "stage2_optimizer_success": bool(getattr(stage2_result, "success", False)),
        "initial_soc_fallback_used": initial_soc_fallback_used,
        "variables": [
            {
                "name": var.name,
                "kind": var.kind,
                "lower": var.lower,
                "upper": var.upper,
                "initial": var.initial,
                "parameter_name": var.parameter_name,
            }
            for var in variables
        ],
    }
    for var in variables:
        best_result[f"var_{var.name}"] = float(best_mapping["variable_values"][f"var_{var.name}"])

    stage1_history_path = args.output_dir / f"{args.cell}_optimization_stage1_history.csv"
    stage2_history_path = args.output_dir / f"{args.cell}_optimization_stage2_history.csv"
    best_path = args.output_dir / f"{args.cell}_best_result.json"
    final_trace_path = args.output_dir / f"{args.cell}_best_voltage_trace_full.csv"
    stage1_history_df.to_csv(stage1_history_path, index=False)
    stage2_history_df.to_csv(stage2_history_path, index=False)
    full_trace.to_csv(final_trace_path, index=False)
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)
        f.write("\n")
    print(f"Saved stage-1 optimization history: {stage1_history_path}")
    print(f"Saved stage-2 optimization history: {stage2_history_path}")
    print(f"Saved final full-profile trace: {final_trace_path}")
    print(f"Saved best result: {best_path}")
    print(
        "Best stage-2 objective (weighted window RMSE) = "
        f"{float(best_result['objective_stage2_fit_V']):.6f} V"
    )
    print(f"Final full-profile RMSE = {float(best_result['rmse_full_profile_V']):.6f} V")
    print(f"Best parameter values: {_format_best_summary(best_result, variables)}")

    modified_path, config_path = write_db_artifacts(
        output_dir=args.output_dir,
        cell_id=str(args.cell),
        best_result=best_result,
        fixed_parameter_overrides=fixed_parameter_overrides,
        original_base_value_map=base_value_map,
        adjusted_base_value_map=adjusted_base_value_map,
        callable_base_scales=callable_base_scales,
    )
    print(f"Saved DB modified-parameters JSON: {modified_path}")
    print(f"Saved optimized cell-config JSON: {config_path}")

    resolved_name_extention = resolve_safe_parameter_name_extention(
        db_path=args.db_path,
        base_parameter_set_name=str(args.parameter_set),
        requested_name_extention=str(args.db_parameter_name_extention),
        modified_parameters_json_path=modified_path,
    )
    db_run_name = str(args.db_run_name) if args.db_run_name else utc_now_run_name()
    run_phase4_append_to_db(
        args=args,
        modified_parameters_json_path=modified_path,
        optimized_cell_config_path=config_path,
        parameter_name_extention=resolved_name_extention,
        run_name=db_run_name,
    )
    print(
        f"Appended optimized simulation run to DB: db={args.db_path}, "
        f"run_name={db_run_name}, name_extention={resolved_name_extention}"
    )
    optimized_run_id = resolve_optimized_run_id(
        db_path=args.db_path,
        cell_id=str(args.cell),
        model_name=str(args.model_name),
        parameter_set=str(args.parameter_set),
        optimized_run_name=db_run_name,
    )
    optimization_config = {
        "cell": str(args.cell),
        "cycle": None if args.cycle is None else str(args.cycle),
        "model_name": str(args.model_name),
        "parameter_set": str(args.parameter_set),
        "solver_mode": str(args.solver_mode),
        "rmse_weights": dict(rmse_weights),
        "tail_downsample_stride_stage1": int(args.tail_downsample_stride_stage1),
        "tail_downsample_stride_stage2": int(args.tail_downsample_stride_stage2),
        "stage1_maxiter": int(args.stage1_maxiter),
        "stage1_popsize": int(args.stage1_popsize),
        "stage1_seed": int(args.stage1_seed),
        "stage1_variable_limit": int(args.stage1_variable_limit),
        "stage1_selected_variables": [var.name for var in stage1_variables],
        "local_method_requested": str(args.local_method),
        "local_method_effective": str(best_result.get("optimizer", args.local_method)),
        "stage2_maxiter": int(args.stage2_maxiter),
        "initial_soc_fallback_used": bool(initial_soc_fallback_used),
        "selected_variables": [
            {
                "name": var.name,
                "kind": var.kind,
                "parameter_name": var.parameter_name,
                "lower": float(var.lower),
                "upper": float(var.upper),
                "initial": float(var.initial),
            }
            for var in variables
        ],
        "fixed_parameter_overrides": fixed_parameter_overrides,
        "artifact_stage1_history_csv": str(stage1_history_path),
        "artifact_stage2_history_csv": str(stage2_history_path),
        "artifact_best_trace_csv": str(final_trace_path),
        "artifact_best_result_json": str(best_path),
        "db_parameter_name_extention_requested": str(args.db_parameter_name_extention),
        "db_parameter_name_extention_resolved": resolved_name_extention,
        "db_run_name": db_run_name,
    }
    upsert_optimization_metadata(
        db_path=args.db_path,
        simulation_run_id=optimized_run_id,
        base_simulation_run_id=(
            int(base_simulation_run_id) if base_simulation_run_id is not None else None
        ),
        optimization_config=optimization_config,
        best_result=best_result,
    )
    print(f"Recorded optimization metadata: simulation_run_id={optimized_run_id}")

    run_ids = resolve_comparison_run_ids(
        db_path=args.db_path,
        cell_id=str(args.cell),
        model_name=str(args.model_name),
        parameter_set=str(args.parameter_set),
        optimized_run_id=optimized_run_id,
    )
    run_phase4_plot_from_db(args=args, simulation_run_ids=run_ids)
    print(f"Phase4 comparison plot generated for run IDs: {run_ids}")


if __name__ == "__main__":
    main()
