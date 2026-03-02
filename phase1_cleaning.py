from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RAW_CSV_PATH = Path("data/raw_cycling_data.csv")
OUTPUT_CSV_PATH = Path("data/phase1_cleaned_data.csv")
PLOT_DIR = Path("outputs/phase1")

PLOT_COLUMNS = ["current_A", "voltage_V", "cycle", "step"]
INTERPOLATE_COLUMNS = ["current_A", "voltage_V"]


def count_missing_values(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].isna().sum()


def consecutive_missing_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}

    for col in columns:
        missing = df[col].isna()
        if not missing.any():
            summary[col] = {
                "num_missing_runs": 0,
                "max_consecutive_missing": 0,
                "run_lengths": [],
            }
            continue

        # Group consecutive boolean values, then extract lengths where value is True.
        run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
        run_lengths = missing.groupby(run_id).sum()
        true_run_lengths = run_lengths[run_lengths > 0].astype(int).tolist()

        summary[col] = {
            "num_missing_runs": len(true_run_lengths),
            "max_consecutive_missing": max(true_run_lengths),
            "run_lengths": true_run_lengths,
        }

    return summary


def plot_timeseries(df: pd.DataFrame, columns: list[str], out_path: Path, mode:str="Raw") -> None:
    fig, axes = plt.subplots(len(columns), 1, figsize=(15, 10), sharex=True)

    for i, col in enumerate(columns):
        axes[i].plot(df["datetime"], df[col], linewidth=0.8)
        axes[i].set_ylabel(col)
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("datetime")
    fig.suptitle(mode + "values over datetime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def interpolate_with_same_step_neighbors(
    df: pd.DataFrame, value_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    out = df.copy()
    n = len(out)
    summary: dict[str, dict[str, int]] = {}

    for col in value_columns:
        summary[col] = {
            "if_both_neighbors": 0,
            "elif_prev_neighbor": 0,
            "elif_next_neighbor": 0,
            "none_no_fill": 0,
            "total_missing_rows_seen": 0,
        }
        missing_idx = out.index[out[col].isna()]
        for idx in missing_idx:
            summary[col]["total_missing_rows_seen"] += 1
            pos = out.index.get_loc(idx)
            step_now = out.at[idx, "step"]

            prev_pos = pos - 1
            next_pos = pos + 1

            prev_ok = prev_pos >= 0 and out.iloc[prev_pos]["step"] == step_now
            next_ok = next_pos < n and out.iloc[next_pos]["step"] == step_now

            prev_val = out.iloc[prev_pos][col] if prev_ok else np.nan
            next_val = out.iloc[next_pos][col] if next_ok else np.nan

            if prev_ok and next_ok and pd.notna(prev_val) and pd.notna(next_val):
                # Linear interpolation between immediate same-step neighbors.
                out.at[idx, col] = (prev_val + next_val) / 2.0
                summary[col]["if_both_neighbors"] += 1
            elif prev_ok and pd.notna(prev_val):
                out.at[idx, col] = prev_val
                summary[col]["elif_prev_neighbor"] += 1
            elif next_ok and pd.notna(next_val):
                out.at[idx, col] = next_val
                summary[col]["elif_next_neighbor"] += 1
            else:
                summary[col]["none_no_fill"] += 1

    return out, summary


def fix_step_short_anomaly_runs(df: pd.DataFrame, max_run_len: int = 4) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    corrected = 0
    steps = out["step"].tolist()
    n = len(steps)
    if n == 0:
        return out, corrected

    runs: list[tuple[int, int, float]] = []
    start = 0
    for pos in range(1, n + 1):
        if pos == n or steps[pos] != steps[start]:
            runs.append((start, pos - 1, steps[start]))
            start = pos

    for run_idx in range(1, len(runs) - 1):
        curr_start, curr_end, curr_val = runs[run_idx]
        prev_start, prev_end, prev_val = runs[run_idx - 1]
        next_start, next_end, next_val = runs[run_idx + 1]
        del prev_start, prev_end, next_start, next_end

        run_len = curr_end - curr_start + 1
        if run_len > max_run_len:
            continue

        if pd.isna(prev_val) or pd.isna(curr_val) or pd.isna(next_val):
            continue

        if prev_val == next_val and curr_val != prev_val:
            out.iloc[curr_start : curr_end + 1, out.columns.get_loc("step")] = prev_val
            corrected += run_len

    return out, corrected


def _extract_step_runs(steps: list[float]) -> list[tuple[int, int, float]]:
    runs: list[tuple[int, int, float]] = []
    if not steps:
        return runs

    start = 0
    for pos in range(1, len(steps) + 1):
        if pos == len(steps) or steps[pos] != steps[start]:
            runs.append((start, pos - 1, steps[start]))
            start = pos
    return runs


def reindex_step_by_cell_protocol(df: pd.DataFrame, cell_id: object) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    changed_rows = 0
    cycle_series = out["cycle"].round().astype(int)

    for cycle_number in cycle_series.drop_duplicates().tolist():
        cycle_mask = cycle_series == cycle_number
        cycle_idx = out.index[cycle_mask].tolist()
        cycle_steps = out.loc[cycle_mask, "step"].astype(int).tolist()
        runs = _extract_step_runs(cycle_steps)
        run_values = [int(run_val) for _, _, run_val in runs]

        # All cells: reindex run-order sequentially from 1.
        mapping = list(range(1, len(runs) + 1))

        for (run_start, run_end, _), new_step in zip(runs, mapping):
            abs_start = cycle_idx[run_start]
            abs_end = cycle_idx[run_end]
            original_values = out.loc[abs_start:abs_end, "step"]
            changed_rows += int((original_values != new_step).sum())
            out.loc[abs_start:abs_end, "step"] = new_step

    return out, changed_rows


def normalize_cycle_jumps_by_shifting_tail(df: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    out = df.copy()
    jump_events = 0
    total_shift = 0.0
    n = len(out)
    if n == 0:
        return out, jump_events, total_shift

    corrected_cycles: list[float] = [float(out.iloc[0]["cycle"])]
    offset = 0.0

    for pos in range(1, n):
        raw_cycle = out.iloc[pos]["cycle"]
        prev_corrected = corrected_cycles[-1]

        if pd.isna(raw_cycle):
            corrected_cycles.append(raw_cycle)
            continue

        adjusted_cycle = float(raw_cycle) - offset
        jump = adjusted_cycle - prev_corrected
        if jump > 1:
            extra = jump - 1
            offset += extra
            total_shift += extra
            adjusted_cycle -= extra
            jump_events += 1

        corrected_cycles.append(adjusted_cycle)

    out["cycle"] = corrected_cycles
    return out, jump_events, total_shift


def print_consecutive_summary(title: str, summary: dict[str, dict[str, object]]) -> None:
    print(f"\n{title}")
    for col, stats in summary.items():
        print(
            f"- {col}: runs={stats['num_missing_runs']}, "
            f"max_consecutive={stats['max_consecutive_missing']}"
        )


def sanitize_filename(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {RAW_CSV_PATH}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 0) Check if datetime has missing values.
    datetime_missing_count = int(df["datetime"].isna().sum())
    print(f"Missing datetime values: {datetime_missing_count}")
    if datetime_missing_count > 0:
        print("Warning: missing/invalid datetime values exist.")
    else:
        print("datetime column has no missing values.")

    if "cell_id" not in df.columns:
        raise KeyError("cell_id column is required for per-cell processing.")

    cleaned_parts: list[pd.DataFrame] = []

    for cell_id, cell_df in df.groupby("cell_id", sort=False):
        cell_df = cell_df.copy()
        cell_tag = sanitize_filename(cell_id)

        print(f"\n=== Processing cell_id: {cell_id} ===")
        print("runs = number of contiguous missing-value segments in time order")

        # 1) Visualize values over datetime without filling missing data (per cell).
        plot_timeseries(cell_df, PLOT_COLUMNS, PLOT_DIR / f"01_raw_timeseries_{cell_tag}.png", mode="Raw")

        # 2) Count missing values (before interpolation, per cell).
        missing_before = count_missing_values(cell_df, PLOT_COLUMNS)
        print("\nMissing values BEFORE interpolation:")
        print(f"Total datapoints: {len(cell_df)}")
        print(missing_before.to_string())

        # 3) Count consecutive missing values (before interpolation, per cell).
        consecutive_before = consecutive_missing_summary(cell_df, PLOT_COLUMNS)
        print_consecutive_summary("Consecutive missing summary BEFORE interpolation:", consecutive_before)

        # 4) Interpolate missing current/voltage when same-step neighbor exists (per cell).
        cleaned_cell_df, interpolation_summary = interpolate_with_same_step_neighbors(
            cell_df, INTERPOLATE_COLUMNS
        )
        print("Interpolation decision summary:")
        for col in INTERPOLATE_COLUMNS:
            stats = interpolation_summary[col]
            print(
                f"- {col}: if={stats['if_both_neighbors']}, "
                f"elif_prev={stats['elif_prev_neighbor']}, "
                f"elif_next={stats['elif_next_neighbor']}, "
                f"none={stats['none_no_fill']}, "
                f"total_missing={stats['total_missing_rows_seen']}"
            )

        # Apply protocol-index corrections per request.
        cleaned_cell_df, step_fixes = fix_step_short_anomaly_runs(cleaned_cell_df, max_run_len=4)
        cleaned_cell_df, cycle_fixes, cycle_total_shift = normalize_cycle_jumps_by_shifting_tail(cleaned_cell_df)
        cleaned_cell_df, step_reindexed_rows = reindex_step_by_cell_protocol(cleaned_cell_df, cell_id)
        print(f"Step corrections applied: {step_fixes}")
        print(f"Cycle jump events corrected: {cycle_fixes}")
        print(f"Total cycle shift applied to tail segments: {cycle_total_shift:g}")
        print(f"Step rows reindexed by protocol mapping: {step_reindexed_rows}")

        # 5) Count missing values again after interpolation and index correction (per cell).
        missing_after = count_missing_values(cleaned_cell_df, PLOT_COLUMNS)
        print("\nMissing values AFTER interpolation:")
        print(missing_after.to_string())

        consecutive_after = consecutive_missing_summary(cleaned_cell_df, PLOT_COLUMNS)
        print_consecutive_summary("Consecutive missing summary AFTER interpolation:", consecutive_after)

        # Additional plot: cleaned values over datetime after interpolation.
        plot_timeseries(cleaned_cell_df, PLOT_COLUMNS, PLOT_DIR / f"02_cleaned_timeseries_{cell_tag}.png", mode="Cleaned")

        cleaned_parts.append(cleaned_cell_df)

    # Concatenate all per-cell cleaned data before writing output.
    cleaned_df = pd.concat(cleaned_parts, ignore_index=True)

    cleaned_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved cleaned CSV: {OUTPUT_CSV_PATH}")
    print(f"Saved plots in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
