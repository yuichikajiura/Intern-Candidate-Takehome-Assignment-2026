from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pybamm

from phase2_simulation import (
    apply_parameter_overrides,
    apply_capacity_scaling,
    get_cell_initial_conditions,
    load_cell_config,
    make_model,
)


DIFFUSION_CANDIDATES = [
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Electrolyte diffusivity [m2.s-1]",
]
REACTION_CANDIDATES = [
    "Negative electrode exchange-current density [A.m-2]",
    "Positive electrode exchange-current density [A.m-2]",
]
OHMIC_CANDIDATES = [
    "Negative electrode conductivity [S.m-1]",
    "Positive electrode conductivity [S.m-1]",
    "Electrolyte conductivity [S.m-1]",
]
SELECTOR_CHOICES = ["n", "p", "e"]
DEFAULT_CELL_CONFIG_JSON_PATH = Path("data/phase2_cell_config.json")


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


def filter_cell_cycles(
    df: pd.DataFrame,
    cell_id: str,
    cycle_range: tuple[int, int] | None,
) -> pd.DataFrame:
    cell_df = df[df["cell_id"] == cell_id].copy()
    if cell_df.empty:
        raise ValueError(f"Cell not found in input data: {cell_id}")
    if cycle_range is not None:
        start, end = cycle_range
        cell_df = cell_df[
            (cell_df["cycle"] >= float(start)) & (cell_df["cycle"] <= float(end))
        ].copy()
    if cell_df.empty:
        raise ValueError(
            f"No rows found for cell={cell_id} after cycle filter {cycle_range}."
        )
    return cell_df.sort_values("datetime").reset_index(drop=True)


def downsample_for_fitting(cell_df: pd.DataFrame, tail_stride: int = 60) -> pd.DataFrame:
    """Keep all transient points and stride only the long tail."""
    if tail_stride <= 1:
        return cell_df.reset_index(drop=True).copy()

    df = cell_df.reset_index(drop=True).copy()
    if "time_in_step_s" in df.columns:
        t = df["time_in_step_s"].to_numpy(dtype=float)
    elif {"time_s", "cycle", "step"}.issubset(df.columns):
        frame = build_step_time_frame(cell_df=df, time_s=df["time_s"].to_numpy(dtype=float))
        t = frame["time_in_step_s"].to_numpy(dtype=float)
    elif {"datetime", "cycle", "step"}.issubset(df.columns):
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.isna().any():
            raise ValueError("Unable to compute time_in_step_s: datetime contains NaT values.")
        t_s = (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        frame = build_step_time_frame(cell_df=df, time_s=t_s)
        t = frame["time_in_step_s"].to_numpy(dtype=float)
    else:
        raise KeyError(
            "downsample_for_fitting requires one of: "
            "'time_in_step_s', or ('time_s','cycle','step'), or "
            "('datetime','cycle','step')."
        )

    keep = np.zeros(len(df), dtype=bool)
    keep[t <= 120.0] = True

    tail_idx = np.flatnonzero(t > 120.0)
    if tail_idx.size > 0:
        keep[tail_idx[:: int(tail_stride)]] = True
        keep[tail_idx[-1]] = True

    return df.loc[keep].reset_index(drop=True)


def get_base_parameter_value(parameter_set: str, parameter_name: str) -> object:
    params = pybamm.ParameterValues(parameter_set)
    if parameter_name not in params.keys():
        raise KeyError(f"Parameter name not found in {parameter_set}: {parameter_name}")
    return params[parameter_name]


def parameter_name_from_selector(category: str, selector: str) -> str:
    s = str(selector).strip().lower()
    if s not in SELECTOR_CHOICES:
        raise ValueError(f"Selector must be one of {SELECTOR_CHOICES}, got: {selector}")
    if category == "ohmic":
        return {
            "n": "Negative electrode conductivity [S.m-1]",
            "p": "Positive electrode conductivity [S.m-1]",
            "e": "Electrolyte conductivity [S.m-1]",
        }[s]
    if category == "kinetic":
        if s == "e":
            raise ValueError(
                "Kinetic selector does not support 'e'. "
                "Use --kinetic-parameter n or --kinetic-parameter p."
            )
        return {
            "n": "Negative electrode exchange-current density [A.m-2]",
            "p": "Positive electrode exchange-current density [A.m-2]",
        }[s]
    if category == "diffusion":
        return {
            "n": "Negative particle diffusivity [m2.s-1]",
            "p": "Positive particle diffusivity [m2.s-1]",
            "e": "Electrolyte diffusivity [m2.s-1]",
        }[s]
    raise ValueError(f"Unknown category: {category}")


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


def scaled_parameter_value(base_value: object, scale: float) -> object:
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


def apply_scaled_override(
    parameter_values: pybamm.ParameterValues,
    parameter_name: str,
    scale: float,
    base_value_map: dict[str, object],
) -> None:
    if parameter_name not in base_value_map:
        raise KeyError(f"Base parameter value not prepared for: {parameter_name}")
    parameter_values.update(
        {parameter_name: scaled_parameter_value(base_value_map[parameter_name], scale)}
    )


def build_step_time_frame(cell_df: pd.DataFrame, time_s: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "cycle": cell_df["cycle"].to_numpy(dtype=float),
            "step": cell_df["step"].to_numpy(dtype=float),
        }
    )
    cycle_changed = frame["cycle"].ne(frame["cycle"].shift(1))
    step_changed = frame["step"].ne(frame["step"].shift(1))
    frame["segment_id"] = (cycle_changed | step_changed).cumsum().astype(int)
    frame["time_in_step_s"] = frame["time_s"] - frame.groupby("segment_id")["time_s"].transform(
        "first"
    )
    return frame


def simulate_voltage_trace(
    cell_df: pd.DataFrame,
    model_name: str,
    parameter_set: str,
    solver_mode: str,
    voltage_max: float | None,
    voltage_min: float | None,
    capacity_ah: float,
    initial_soc: float,
    parameter_overrides: dict[str, object] | None,
    parameter_scales: dict[str, float],
    base_value_map: dict[str, object],
) -> pd.DataFrame:
    t_s = (cell_df["datetime"] - cell_df["datetime"].iloc[0]).dt.total_seconds().to_numpy(
        dtype=float
    )
    i_exp = cell_df["current_A"].to_numpy(dtype=float)
    model = make_model(model_name, parameter_set=parameter_set)
    parameter_values = pybamm.ParameterValues(parameter_set)
    apply_capacity_scaling(parameter_values, float(capacity_ah))
    if voltage_max is not None:
        parameter_values.update({"Upper voltage cut-off [V]": float(voltage_max)})
    if voltage_min is not None:
        parameter_values.update({"Lower voltage cut-off [V]": float(voltage_min)})
    apply_parameter_overrides(
        parameter_values=parameter_values,
        parameter_overrides=parameter_overrides,
    )
    for parameter_name, scale in parameter_scales.items():
        apply_scaled_override(
            parameter_values=parameter_values,
            parameter_name=parameter_name,
            scale=float(scale),
            base_value_map=base_value_map,
        )
    parameter_values.update({"Current function [A]": pybamm.Interpolant(t_s, i_exp, pybamm.t)})
    sim = pybamm.Simulation(
        model=model,
        parameter_values=parameter_values,
        solver=pybamm.CasadiSolver(mode=solver_mode),
    )
    try:
        solution = sim.solve(t_eval=t_s, initial_soc=float(initial_soc))
    except Exception:
        if "composite" not in parameter_set.lower():
            raise
        solution = sim.solve(t_eval=t_s)
    v_sim = np.asarray(solution["Terminal voltage [V]"].entries, dtype=float)
    if len(v_sim) != len(t_s):
        raise RuntimeError(
            "Simulation terminated before full profile completion: "
            f"sim_points={len(v_sim)}, exp_points={len(t_s)}"
        )
    frame = build_step_time_frame(cell_df, t_s)
    frame["voltage_sim_V"] = v_sim
    return frame


def build_rmse_weights(raw_weights: list[float]) -> dict[str, float]:
    if len(raw_weights) != 4:
        raise ValueError("--rmse-weights requires 4 values.")
    if any(float(w) < 0 for w in raw_weights):
        raise ValueError("--rmse-weights must be non-negative.")
    if sum(float(w) for w in raw_weights) <= 0:
        raise ValueError("Sum of --rmse-weights must be > 0.")
    return {
        "ohmic": float(raw_weights[0]),
        "kinetic": float(raw_weights[1]),
        "diffusion": float(raw_weights[2]),
        "capacity": float(raw_weights[3]),
    }


def validate_base_inputs(capacity_ah: float, initial_soc: float) -> None:
    if float(capacity_ah) <= 0:
        raise ValueError("--capacity-ah must be > 0.")
    if not (0.0 <= float(initial_soc) <= 1.0):
        raise ValueError("--initial-soc must be within [0, 1].")


def resolve_capacity_and_initial_soc(
    cell_config_json: Path,
    cell_id: str,
    capacity_ah: float | None,
    initial_soc: float | None,
) -> tuple[float, float]:
    if (capacity_ah is None) != (initial_soc is None):
        raise ValueError(
            "Provide both --capacity-ah and --initial-soc together, or omit both "
            "to load values from --cell-config-json."
        )
    config_map = load_cell_config(cell_config_json)
    resolved_capacity_ah, resolved_initial_soc = get_cell_initial_conditions(
        config_map=config_map,
        cell_id=cell_id,
        fallback_capacity_ah=None if capacity_ah is None else float(capacity_ah),
        fallback_initial_soc=None if initial_soc is None else float(initial_soc),
    )
    return float(resolved_capacity_ah), float(resolved_initial_soc)


def prepare_base_parameter_context(
    parameter_set: str,
    candidate_names: set[str],
) -> tuple[dict[str, object], list[str]]:
    base_value_map: dict[str, object] = {}
    missing_names: list[str] = []
    for name in sorted(candidate_names):
        try:
            base_value_map[name] = get_base_parameter_value(parameter_set, name)
        except KeyError:
            missing_names.append(name)
    return base_value_map, missing_names


def print_phase5_context(
    target_label: str,
    cell_id: str,
    cycle_range: tuple[int, int] | None,
    model_name: str,
    parameter_set: str,
    n_points: int,
    rmse_weights: dict[str, float],
    base_value_map: dict[str, object],
    missing_names: list[str],
) -> None:
    print(
        f"{target_label}: cell={cell_id}, cycles={cycle_range}, model={model_name}, "
        f"parameter_set={parameter_set}, points={n_points}"
    )
    print(
        "Window weights (ohmic, kinetic, diffusion, capacity) = "
        f"{rmse_weights['ohmic']}, {rmse_weights['kinetic']}, "
        f"{rmse_weights['diffusion']}, {rmse_weights['capacity']}"
    )
    for name in sorted(base_value_map):
        value = base_value_map[name]
        print(f"Parameter type: {name} -> {type(value).__name__}, callable={callable(value)}")
    if missing_names:
        print(
            "Warning: parameter(s) missing in selected parameter set and skipped: "
            f"{missing_names}"
        )
