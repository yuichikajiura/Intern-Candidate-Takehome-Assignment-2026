from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pybamm

from phase2_simulation import apply_parameter_overrides, make_model
from phase5_common import build_step_time_frame


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


def _scaled_runtime_value(base_value: object, input_name: str) -> object:
    scale_input = pybamm.InputParameter(input_name)
    if callable(base_value):
        def wrapped(*args: object, _base=base_value, _scale=scale_input) -> object:
            return _scale * _base(*args)

        return wrapped
    if _is_numeric_scalar(base_value):
        scalar = base_value.item() if hasattr(base_value, "item") else base_value
        return float(scalar) * scale_input
    raise ValueError(
        "Unsupported parameter type for runtime scaling. "
        f"type={type(base_value).__name__}"
    )


@dataclass
class CachedSimulationRunner:
    model_name: str
    parameter_set: str
    solver_mode: str
    voltage_max: float | None
    voltage_min: float | None
    base_value_map: dict[str, object]
    time_s_eval: np.ndarray
    current_profile_time_s: np.ndarray
    current_profile_a: np.ndarray
    eval_df: pd.DataFrame
    fixed_parameter_overrides: dict[str, object]
    optimized_parameter_names: list[str]

    _simulation: pybamm.Simulation | None = field(default=None, init=False, repr=False)
    _nominal_capacity_key: str = field(
        default="Nominal cell capacity [A.h]", init=False, repr=False
    )
    _parallel_key: str = field(
        default="Number of electrodes connected in parallel to make a cell",
        init=False,
        repr=False,
    )
    _warned_initial_soc_fallback: bool = field(default=False, init=False, repr=False)
    initial_soc_fallback_used: bool = field(default=False, init=False)

    def build(self) -> None:
        model = make_model(self.model_name, parameter_set=self.parameter_set)
        parameter_values = pybamm.ParameterValues(self.parameter_set)

        if self.voltage_max is not None:
            parameter_values.update({"Upper voltage cut-off [V]": float(self.voltage_max)})
        if self.voltage_min is not None:
            parameter_values.update({"Lower voltage cut-off [V]": float(self.voltage_min)})

        # Apply fixed overrides first, then assign runtime inputs for optimized quantities.
        apply_parameter_overrides(
            parameter_values=parameter_values,
            parameter_overrides=self.fixed_parameter_overrides,
        )

        base_nominal_capacity = float(parameter_values[self._nominal_capacity_key])
        if base_nominal_capacity <= 0:
            raise ValueError(f"Invalid base nominal capacity: {base_nominal_capacity}")
        capacity_input = pybamm.InputParameter("capacity_ah")
        updates: dict[str, object] = {self._nominal_capacity_key: capacity_input}

        if self._parallel_key in parameter_values.keys():
            base_parallel = float(parameter_values[self._parallel_key])
            updates[self._parallel_key] = (base_parallel / base_nominal_capacity) * capacity_input
        parameter_values.update(updates)

        for parameter_name in self.optimized_parameter_names:
            if parameter_name not in self.base_value_map:
                raise KeyError(f"Missing base parameter value: {parameter_name}")
            input_name = f"scale::{parameter_name}"
            parameter_values.update(
                {parameter_name: _scaled_runtime_value(self.base_value_map[parameter_name], input_name)}
            )

        parameter_values.update(
            {
                "Current function [A]": pybamm.Interpolant(
                    self.current_profile_time_s,
                    self.current_profile_a,
                    pybamm.t,
                )
            }
        )
        solver = pybamm.CasadiSolver(mode=self.solver_mode)
        self._simulation = pybamm.Simulation(
            model=model,
            parameter_values=parameter_values,
            solver=solver,
            output_variables=["Terminal voltage [V]"],
        )

    def solve(self, inputs: dict[str, float]) -> pd.DataFrame:
        if self._simulation is None:
            raise RuntimeError("CachedSimulationRunner must be built before solve().")

        runtime_inputs: dict[str, float] = {
            "capacity_ah": float(inputs["capacity_ah"]),
        }
        for parameter_name in self.optimized_parameter_names:
            key = f"scale::{parameter_name}"
            runtime_inputs[key] = float(inputs.get(key, 1.0))

        initial_soc = float(inputs["initial_soc"])
        try:
            solution = self._simulation.solve(
                t_eval=self.time_s_eval,
                initial_soc=initial_soc,
                inputs=runtime_inputs,
            )
        except Exception:
            if "composite" not in self.parameter_set.lower():
                raise
            if not self._warned_initial_soc_fallback:
                print(
                    "Warning: initial_soc solve path failed for composite parameter set; "
                    "falling back to solve without initial_soc. initial_soc optimization may "
                    "be less effective under this fallback."
                )
                self._warned_initial_soc_fallback = True
            self.initial_soc_fallback_used = True
            solution = self._simulation.solve(
                t_eval=self.time_s_eval,
                inputs=runtime_inputs,
            )

        voltage = np.asarray(solution["Terminal voltage [V]"].entries, dtype=float)
        if len(voltage) != len(self.time_s_eval):
            raise RuntimeError(
                "Simulation terminated before full profile completion: "
                f"sim_points={len(voltage)}, exp_points={len(self.time_s_eval)}"
            )
        frame = build_step_time_frame(self.eval_df, self.time_s_eval)
        frame["voltage_sim_V"] = voltage
        frame["voltage_exp_V"] = self.eval_df["voltage_V"].to_numpy(dtype=float)
        frame["error_V"] = frame["voltage_sim_V"] - frame["voltage_exp_V"]
        return frame


def build_cached_runner(
    *,
    model_name: str,
    parameter_set: str,
    solver_mode: str,
    voltage_max: float | None,
    voltage_min: float | None,
    base_value_map: dict[str, object],
    current_profile_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    fixed_parameter_overrides: dict[str, object],
    optimized_parameter_names: list[str],
) -> CachedSimulationRunner:
    t0 = current_profile_df["datetime"].iloc[0]
    current_t_s = (current_profile_df["datetime"] - t0).dt.total_seconds().to_numpy(
        dtype=float
    )
    eval_t_s = (eval_df["datetime"] - t0).dt.total_seconds().to_numpy(
        dtype=float
    )
    i_exp = current_profile_df["current_A"].to_numpy(dtype=float)
    runner = CachedSimulationRunner(
        model_name=model_name,
        parameter_set=parameter_set,
        solver_mode=solver_mode,
        voltage_max=voltage_max,
        voltage_min=voltage_min,
        base_value_map=base_value_map,
        time_s_eval=eval_t_s,
        current_profile_time_s=current_t_s,
        current_profile_a=i_exp,
        eval_df=eval_df.reset_index(drop=True).copy(),
        fixed_parameter_overrides=fixed_parameter_overrides,
        optimized_parameter_names=optimized_parameter_names,
    )
    runner.build()
    return runner


def solve_with_inputs(
    *,
    runner: CachedSimulationRunner,
    inputs: dict[str, float],
) -> pd.DataFrame:
    return runner.solve(inputs=inputs)
