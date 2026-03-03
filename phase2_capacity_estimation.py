from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
OUTPUT_DIR = Path("outputs/phase2")


@dataclass(frozen=True)
class CapacityWindow:
    cell_id: str
    cycle: int
    step: int
    label: str


CAPACITY_WINDOWS = [
    CapacityWindow("CELL_A", 1, 5, "first-cycle CC discharge"),
    CapacityWindow("CELL_B", 1, 7, "first-cycle fast CC discharge"),
    CapacityWindow("CELL_B", 2, 2, "second-cycle slow CC discharge"),
    CapacityWindow("CELL_C", 2, 3, "second-cycle CC discharge"),
    CapacityWindow("CELL_D", 2, 1, "second-cycle CC charge"),
    CapacityWindow("CELL_D", 2, 2, "second-cycle CV charge"),
    CapacityWindow("CELL_E", 2, 5, "second-cycle CC discharge"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-cell capacity from selected cycle/step windows "
            "by integrating current over time."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV_PATH,
        help="Path to cleaned Phase 1 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for capacity estimation outputs.",
    )
    parser.add_argument(
        "--default-initial-soc",
        type=float,
        default=0.95,
        help="Default initial SoC written to generated cell config JSON.",
    )
    parser.add_argument(
        "--capacity-round-decimals",
        type=int,
        default=4,
        help="Round estimated capacities in generated JSON.",
    )
    return parser.parse_args()


def load_cleaned_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"datetime", "cell_id", "cycle", "step", "current_A"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {input_csv}: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "cell_id", "cycle", "step", "current_A"]).copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["cycle", "step"]).copy()
    df = df.sort_values(["cell_id", "datetime"]).reset_index(drop=True)
    return df


def integrate_capacity_ah(segment: pd.DataFrame) -> tuple[float, float]:
    times_s = (
        segment["datetime"] - segment["datetime"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float)
    current_a = segment["current_A"].to_numpy(dtype=float)

    if len(times_s) < 2:
        raise ValueError("Need at least 2 points to integrate capacity.")

    # NumPy >=2 uses trapezoid, while older versions still provide trapz.
    if hasattr(np, "trapezoid"):
        signed_as = np.trapezoid(current_a, times_s)
    else:
        signed_as = np.trapz(current_a, times_s)
    signed_ah = float(signed_as / 3600.0)
    abs_ah = abs(signed_ah)
    return signed_ah, abs_ah


def extract_window(df: pd.DataFrame, window: CapacityWindow) -> pd.DataFrame:
    is_cell = df["cell_id"] == window.cell_id
    is_cycle = np.isclose(df["cycle"], window.cycle, atol=1e-9)
    is_step = np.isclose(df["step"], window.step, atol=1e-9)
    seg = df[is_cell & is_cycle & is_step].copy()
    seg = seg.sort_values("datetime")
    if seg.empty:
        raise ValueError(
            f"No data found for {window.cell_id} cycle={window.cycle} step={window.step}."
        )
    return seg


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.default_initial_soc <= 1.0):
        raise ValueError("--default-initial-soc must be in [0, 1].")

    df = load_cleaned_data(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    details_rows: list[dict[str, object]] = []
    per_cell_abs_sum: dict[str, float] = {}
    per_cell_windows_count: dict[str, int] = {}

    for window in CAPACITY_WINDOWS:
        seg = extract_window(df, window)
        signed_ah, abs_ah = integrate_capacity_ah(seg)
        details_rows.append(
            {
                "cell_id": window.cell_id,
                "cycle": window.cycle,
                "step": window.step,
                "label": window.label,
                "points": len(seg),
                "start_datetime": seg["datetime"].iloc[0],
                "end_datetime": seg["datetime"].iloc[-1],
                "signed_capacity_ah": signed_ah,
                "abs_capacity_ah": abs_ah,
            }
        )
        per_cell_abs_sum[window.cell_id] = per_cell_abs_sum.get(window.cell_id, 0.0) + abs_ah
        per_cell_windows_count[window.cell_id] = per_cell_windows_count.get(window.cell_id, 0) + 1

    detail_df = pd.DataFrame(details_rows)
    detail_df.to_csv(args.output_dir / "capacity_window_details.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    cell_config: dict[str, dict[str, float]] = {}
    for cell_id in sorted(per_cell_abs_sum):
        estimated_capacity = per_cell_abs_sum[cell_id]
        summary_rows.append(
            {
                "cell_id": cell_id,
                "aggregation": "sum_abs_selected_windows",
                "num_windows": per_cell_windows_count[cell_id],
                "estimated_capacity_ah": estimated_capacity,
            }
        )
        cell_config[cell_id] = {
            "capacity_ah": round(estimated_capacity, args.capacity_round_decimals),
            "initial_soc": float(args.default_initial_soc),
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "capacity_estimates.csv", index=False)

    config_path = args.output_dir / "phase2_cell_config_from_capacity.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cell_config, f, indent=2)
        f.write("\n")

    print(f"Saved: {args.output_dir / 'capacity_window_details.csv'}")
    print(f"Saved: {args.output_dir / 'capacity_estimates.csv'}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
