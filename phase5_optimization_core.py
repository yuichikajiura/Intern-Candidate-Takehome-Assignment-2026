from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phase5_cached_runner import CachedSimulationRunner, solve_with_inputs

try:
    from scipy.optimize import differential_evolution, minimize
except Exception:  # pragma: no cover - runtime guard
    differential_evolution = None
    minimize = None


@dataclass(frozen=True)
class OptimizationVariable:
    name: str
    kind: str
    lower: float
    upper: float
    initial: float
    parameter_name: str | None = None


def _rmse_for_mask(trace: pd.DataFrame, mask: pd.Series) -> float:
    vals = trace.loc[mask, "error_V"].to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(vals))))


def _capacity_tail_mask(trace: pd.DataFrame, tail_stride: int) -> np.ndarray:
    t = trace["time_in_step_s"].to_numpy(dtype=float)
    base_mask = t >= 120.0
    if int(tail_stride) <= 1:
        return base_mask

    # Downsample only the long-tail (120s+) points, independently per step segment.
    keep_idx: list[object] = []
    tail_df = trace.loc[base_mask, ["segment_id"]]
    for _, segment_df in tail_df.groupby("segment_id", sort=False):
        idx = segment_df.index.to_numpy()
        if idx.size == 0:
            continue
        selected = idx[:: int(tail_stride)]
        if selected.size == 0 or selected[-1] != idx[-1]:
            selected = np.append(selected, idx[-1])
        keep_idx.extend(selected.tolist())
    return trace.index.isin(keep_idx)


def evaluate_window_rmses(
    trace: pd.DataFrame,
    *,
    capacity_tail_stride: int = 1,
) -> dict[str, float]:
    t = trace["time_in_step_s"]
    capacity_mask = _capacity_tail_mask(trace=trace, tail_stride=int(capacity_tail_stride))
    return {
        "rmse_ohmic_0_2s_V": _rmse_for_mask(trace, (t >= 0.0) & (t < 2.0)),
        "rmse_kinetic_2_20s_V": _rmse_for_mask(trace, (t >= 2.0) & (t < 20.0)),
        "rmse_diffusion_20_120s_V": _rmse_for_mask(trace, (t >= 20.0) & (t < 120.0)),
        "rmse_capacity_120plus_s_V": _rmse_for_mask(trace, capacity_mask),
        "rmse_full_profile_V": float(
            np.sqrt(np.mean(np.square(trace["error_V"].to_numpy(dtype=float))))
        ),
    }


def weighted_window_rmse(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> float:
    pairs = [
        ("rmse_ohmic_0_2s_V", float(weights["ohmic"])),
        ("rmse_kinetic_2_20s_V", float(weights["kinetic"])),
        ("rmse_diffusion_20_120s_V", float(weights["diffusion"])),
        ("rmse_capacity_120plus_s_V", float(weights["capacity"])),
    ]
    weighted_sum = 0.0
    weight_sum = 0.0
    for metric_name, w in pairs:
        value = float(metrics[metric_name])
        if not np.isfinite(value):
            continue
        if w < 0:
            raise ValueError("RMSE weights must be non-negative.")
        weighted_sum += w * value
        weight_sum += w
    if weight_sum <= 0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def vector_to_parameter_dict(
    x: np.ndarray,
    variables: list[OptimizationVariable],
    baseline_capacity_ah: float,
    baseline_initial_soc: float,
) -> dict[str, object]:
    capacity_ah = float(baseline_capacity_ah)
    initial_soc = float(baseline_initial_soc)
    parameter_scales: dict[str, float] = {}
    variable_values: dict[str, float] = {}

    for value, var in zip(x, variables):
        v = float(value)
        variable_values[f"var_{var.name}"] = v
        if var.kind == "initial_soc":
            initial_soc = v
        elif var.kind == "capacity_scale":
            capacity_ah = float(baseline_capacity_ah) * v
        elif var.kind == "parameter_scale":
            if var.parameter_name is None:
                raise ValueError(f"Missing parameter_name for variable: {var.name}")
            parameter_scales[var.parameter_name] = v
        else:
            raise ValueError(f"Unsupported variable kind: {var.kind}")

    return {
        "capacity_ah": float(capacity_ah),
        "initial_soc": float(initial_soc),
        "parameter_scales": parameter_scales,
        "variable_values": variable_values,
    }


def objective_factory(
    *,
    stage_label: str,
    runner: CachedSimulationRunner,
    variables: list[OptimizationVariable],
    baseline_capacity_ah: float,
    baseline_initial_soc: float,
    rmse_weights: dict[str, float],
    penalty_rmse: float,
    capacity_tail_stride: int,
    history: list[dict[str, object]],
):
    def objective(x: np.ndarray) -> float:
        mapped = vector_to_parameter_dict(
            x=np.asarray(x, dtype=float),
            variables=variables,
            baseline_capacity_ah=baseline_capacity_ah,
            baseline_initial_soc=baseline_initial_soc,
        )

        capacity_ah = float(mapped["capacity_ah"])
        initial_soc = float(mapped["initial_soc"])
        parameter_scales = dict(mapped["parameter_scales"])
        variable_values = dict(mapped["variable_values"])

        inputs: dict[str, float] = {
            "capacity_ah": capacity_ah,
            "initial_soc": initial_soc,
        }
        for parameter_name, scale in parameter_scales.items():
            inputs[f"scale::{parameter_name}"] = float(scale)

        status = "ok"
        message = ""
        try:
            trace = solve_with_inputs(runner=runner, inputs=inputs)
            metrics = evaluate_window_rmses(
                trace,
                capacity_tail_stride=int(capacity_tail_stride),
            )
            objective_v = weighted_window_rmse(metrics=metrics, weights=rmse_weights)
            if not np.isfinite(objective_v):
                objective_v = float(penalty_rmse)
        except Exception as exc:
            status = "failed"
            message = str(exc)
            objective_v = float(penalty_rmse)
            metrics = {
                "rmse_ohmic_0_2s_V": float(penalty_rmse),
                "rmse_kinetic_2_20s_V": float(penalty_rmse),
                "rmse_diffusion_20_120s_V": float(penalty_rmse),
                "rmse_capacity_120plus_s_V": float(penalty_rmse),
                "rmse_full_profile_V": float(penalty_rmse),
            }

        row: dict[str, object] = {
            "stage": stage_label,
            "objective_V": float(objective_v),
            "status": status,
            "message": message,
            "capacity_ah": capacity_ah,
            "initial_soc": initial_soc,
            **metrics,
            **variable_values,
        }
        for var in variables:
            if var.parameter_name is not None:
                row[f"parameter_{var.name}"] = var.parameter_name
        history.append(row)
        return float(objective_v)

    return objective


def _best_row(history: list[dict[str, object]]) -> dict[str, object]:
    if not history:
        return {"objective_V": float("inf")}
    finite = [row for row in history if np.isfinite(float(row["objective_V"]))]
    if not finite:
        return dict(history[-1])
    return dict(min(finite, key=lambda row: float(row["objective_V"])))


def run_coarse_global_optimization(
    *,
    runner: CachedSimulationRunner,
    variables: list[OptimizationVariable],
    baseline_capacity_ah: float,
    baseline_initial_soc: float,
    rmse_weights: dict[str, float],
    penalty_rmse: float,
    capacity_tail_stride: int,
    stage1_maxiter: int,
    stage1_popsize: int,
    stage1_seed: int,
) -> tuple[dict[str, object], pd.DataFrame, object]:
    if differential_evolution is None:
        raise RuntimeError("scipy is required for phase5 optimization.")
    if len(variables) == 0:
        raise ValueError("No optimization variables were selected.")

    history: list[dict[str, object]] = []
    objective = objective_factory(
        stage_label="stage1",
        runner=runner,
        variables=variables,
        baseline_capacity_ah=baseline_capacity_ah,
        baseline_initial_soc=baseline_initial_soc,
        rmse_weights=rmse_weights,
        penalty_rmse=penalty_rmse,
        capacity_tail_stride=int(capacity_tail_stride),
        history=history,
    )

    bounds = [(var.lower, var.upper) for var in variables]
    x0 = np.asarray([float(var.initial) for var in variables], dtype=float)
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=int(stage1_maxiter),
        popsize=int(stage1_popsize),
        seed=int(stage1_seed),
        polish=False,
        updating="deferred",
        workers=1,
        x0=x0,
    )
    best = _best_row(history)
    best["optimizer"] = "differential_evolution"
    best["optimizer_success"] = bool(result.success)
    best["optimizer_message"] = str(result.message)
    best["optimizer_nit"] = int(getattr(result, "nit", -1))
    best["optimizer_nfev"] = int(getattr(result, "nfev", len(history)))
    best["x"] = [float(v) for v in np.asarray(result.x, dtype=float)]
    return best, pd.DataFrame(history), result


def _needs_fallback(
    result: object,
    history: list[dict[str, object]],
    x0: np.ndarray,
) -> bool:
    success = bool(getattr(result, "success", False))
    fun = float(getattr(result, "fun", float("nan")))
    if not success or not np.isfinite(fun):
        return True
    if len(history) <= 1:
        return True
    start = float(history[0]["objective_V"])
    best = float(min(row["objective_V"] for row in history))
    if not np.isfinite(start) or not np.isfinite(best):
        return True
    absolute_improvement = abs(start - best)
    relative_improvement = absolute_improvement / max(abs(start), 1e-8)
    # Treat tiny practical improvement as local stalling.
    return (
        absolute_improvement < 1e-5
        or relative_improvement < 1e-4
    ) and np.allclose(np.asarray(getattr(result, "x", x0)), x0)


def run_local_refinement(
    *,
    runner: CachedSimulationRunner,
    variables: list[OptimizationVariable],
    baseline_capacity_ah: float,
    baseline_initial_soc: float,
    rmse_weights: dict[str, float],
    penalty_rmse: float,
    capacity_tail_stride: int,
    x0: np.ndarray,
    local_method: str,
    stage2_maxiter: int,
) -> tuple[dict[str, object], pd.DataFrame, object]:
    if minimize is None:
        raise RuntimeError("scipy is required for phase5 optimization.")

    def _run(method: str) -> tuple[dict[str, object], pd.DataFrame, object]:
        history: list[dict[str, object]] = []
        objective = objective_factory(
            stage_label="stage2",
            runner=runner,
            variables=variables,
            baseline_capacity_ah=baseline_capacity_ah,
            baseline_initial_soc=baseline_initial_soc,
            rmse_weights=rmse_weights,
            penalty_rmse=penalty_rmse,
            capacity_tail_stride=int(capacity_tail_stride),
            history=history,
        )
        bounds = [(var.lower, var.upper) for var in variables]
        result = minimize(
            objective,
            x0=np.asarray(x0, dtype=float),
            method=method,
            bounds=bounds,
            options={"maxiter": int(stage2_maxiter)},
        )
        best = _best_row(history)
        best["optimizer"] = method
        best["optimizer_success"] = bool(result.success)
        best["optimizer_message"] = str(result.message)
        best["optimizer_nit"] = int(getattr(result, "nit", -1))
        best["optimizer_nfev"] = int(getattr(result, "nfev", len(history)))
        best["x"] = [float(v) for v in np.asarray(getattr(result, "x", x0), dtype=float)]
        return best, pd.DataFrame(history), result

    requested = str(local_method)
    stage2_best, stage2_history, stage2_result = _run(requested)
    stage2_history["local_method_attempt"] = requested
    if requested == "L-BFGS-B" and _needs_fallback(
        stage2_result, stage2_history.to_dict("records"), np.asarray(x0, dtype=float)
    ):
        fallback_best, fallback_history, fallback_result = _run("Powell")
        fallback_history["local_method_attempt"] = "Powell_fallback"
        combined_history = pd.concat([stage2_history, fallback_history], ignore_index=True)
        fallback_best["fallback_from"] = requested
        return fallback_best, combined_history, fallback_result
    return stage2_best, stage2_history, stage2_result
