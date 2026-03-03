from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
OUTPUT_DIR = Path("outputs/phase2")


# Step definitions are aligned with phase2_capacity_estimation.py
CELL_STEPS = {
    "CELL_A": [5],      # CC discharge
    "CELL_B": [7],      # fast CC discharge
    "CELL_C": [3],      # CC discharge
    "CELL_D": [1, 2],   # CC + CV charge
    "CELL_E": [5],      # CC discharge
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate capacity decay from selected cycle-step windows per cell. "
            "Uses first/last cycle for most cells, with cycle-2 baseline for CELL_D."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cells", nargs="+", default=None, help="Optional subset of cells.")
    return parser.parse_args()


def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"datetime", "cell_id", "cycle", "step", "current_A"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["current_A"] = pd.to_numeric(df["current_A"], errors="coerce")
    df = df.dropna(subset=["datetime", "cell_id", "cycle", "step", "current_A"]).copy()
    df = df.sort_values(["cell_id", "datetime"]).reset_index(drop=True)
    return df


def integrate_capacity_ah(segment: pd.DataFrame) -> tuple[float, float]:
    if len(segment) < 2:
        raise ValueError("Need at least 2 points to integrate capacity.")
    t_s = (segment["datetime"] - segment["datetime"].iloc[0]).dt.total_seconds().to_numpy(float)
    i_a = segment["current_A"].to_numpy(float)
    signed_as = np.trapezoid(i_a, t_s) if hasattr(np, "trapezoid") else np.trapz(i_a, t_s)
    signed_ah = float(signed_as / 3600.0)
    return signed_ah, abs(signed_ah)


def pick_reference_cycles(cycles: list[int], cell_id: str) -> tuple[list[int], list[int]]:
    if not cycles:
        raise ValueError(f"No cycles found for {cell_id}.")
    unique_cycles = sorted(set(cycles))

    if cell_id == "CELL_D":
        # Use cycle 2 as baseline (before degradation) and compare to final cycle.
        if len(unique_cycles) < 2:
            raise ValueError(f"{cell_id} requires at least 2 cycles; found {len(unique_cycles)}.")
        if 2 not in unique_cycles:
            raise ValueError(f"{cell_id} baseline cycle 2 not found in available cycles: {unique_cycles}")
        first_cycles = [2]
        last_cycles = [unique_cycles[-1]]
    else:
        if len(unique_cycles) < 2:
            raise ValueError(f"{cell_id} requires at least 2 cycles; found {len(unique_cycles)}.")
        first_cycles = [unique_cycles[0]]
        last_cycles = [unique_cycles[-1]]
    return first_cycles, last_cycles


def cycle_capacity_from_steps(cell_df: pd.DataFrame, cycle: int, steps: list[int]) -> tuple[float, list[dict[str, object]]]:
    cycle_df = cell_df[np.isclose(cell_df["cycle"].to_numpy(float), float(cycle), atol=1e-9)]
    if cycle_df.empty:
        raise ValueError(f"No rows found for cycle={cycle}.")

    total_abs_ah = 0.0
    details: list[dict[str, object]] = []
    for step in steps:
        step_df = cycle_df[np.isclose(cycle_df["step"].to_numpy(float), float(step), atol=1e-9)].copy()
        if step_df.empty:
            details.append(
                {
                    "cycle": cycle,
                    "step": step,
                    "points": 0,
                    "signed_capacity_ah": np.nan,
                    "abs_capacity_ah": np.nan,
                }
            )
            continue

        step_df = step_df.sort_values("datetime")
        signed_ah, abs_ah = integrate_capacity_ah(step_df)
        total_abs_ah += abs_ah
        details.append(
            {
                "cycle": cycle,
                "step": step,
                "points": len(step_df),
                "signed_capacity_ah": signed_ah,
                "abs_capacity_ah": abs_ah,
            }
        )
    return total_abs_ah, details


def main() -> None:
    args = parse_args()
    df = load_data(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_cells = args.cells if args.cells else sorted(CELL_STEPS.keys())

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for cell_id in selected_cells:
        if cell_id not in CELL_STEPS:
            raise ValueError(f"Unsupported cell_id in script mapping: {cell_id}")

        cell_df = df[df["cell_id"] == cell_id].copy()
        if cell_df.empty:
            raise ValueError(f"No data found for {cell_id}")

        cycles = [int(round(c)) for c in cell_df["cycle"].dropna().unique().tolist()]
        first_cycles, last_cycles = pick_reference_cycles(cycles, cell_id)
        steps = CELL_STEPS[cell_id]

        first_cycle_caps: list[float] = []
        last_cycle_caps: list[float] = []

        for cyc in first_cycles:
            cap, rows = cycle_capacity_from_steps(cell_df, cyc, steps)
            first_cycle_caps.append(cap)
            for r in rows:
                detail_rows.append(
                    {
                        "cell_id": cell_id,
                        "window": "first",
                        **r,
                        "cycle_total_abs_capacity_ah": cap,
                    }
                )

        for cyc in last_cycles:
            cap, rows = cycle_capacity_from_steps(cell_df, cyc, steps)
            last_cycle_caps.append(cap)
            for r in rows:
                detail_rows.append(
                    {
                        "cell_id": cell_id,
                        "window": "last",
                        **r,
                        "cycle_total_abs_capacity_ah": cap,
                    }
                )

        first_ref_ah = float(np.nanmean(first_cycle_caps))
        last_ref_ah = float(np.nanmean(last_cycle_caps))
        decay_ah = first_ref_ah - last_ref_ah
        decay_pct = (decay_ah / first_ref_ah * 100.0) if first_ref_ah > 0 else np.nan

        summary_rows.append(
            {
                "cell_id": cell_id,
                "steps_used": ",".join(str(s) for s in steps),
                "first_cycles": ",".join(str(c) for c in first_cycles),
                "last_cycles": ",".join(str(c) for c in last_cycles),
                "first_reference_capacity_ah": first_ref_ah,
                "last_reference_capacity_ah": last_ref_ah,
                "capacity_decay_ah": decay_ah,
                "capacity_decay_percent": decay_pct,
            }
        )

    detail_path = args.output_dir / "capacity_decay_details.csv"
    summary_path = args.output_dir / "capacity_decay_summary.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
