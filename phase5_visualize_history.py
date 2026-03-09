from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize RMSE/objective evolution across optimization stage 1 and stage 2."
    )
    parser.add_argument(
        "--stage1",
        type=Path,
        required=True,
        help="Path to stage 1 history CSV.",
    )
    parser.add_argument(
        "--stage2",
        type=Path,
        required=True,
        help="Path to stage 2 history CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def load_history(path: Path, stage_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    required = {"objective_V", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["stage"] = stage_name
    df["eval_in_stage"] = np.arange(1, len(df) + 1)
    df["is_ok"] = df["status"].eq("ok")
    df["objective_ok_V"] = df["objective_V"].where(df["is_ok"], np.nan)
    df["running_best_ok_V"] = df["objective_ok_V"].cummin()

    return df


def build_combined_history(stage1: pd.DataFrame, stage2: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([stage1, stage2], ignore_index=True).copy()
    combined["eval_global"] = np.arange(1, len(combined) + 1)
    combined["objective_ok_V"] = combined["objective_V"].where(combined["is_ok"], np.nan)
    combined["running_best_global_ok_V"] = combined["objective_ok_V"].cummin()
    return combined


def summarize_stage(df: pd.DataFrame) -> str:
    n_total = len(df)
    n_ok = int(df["is_ok"].sum())
    n_fail = n_total - n_ok
    best = df["objective_ok_V"].min()
    best_text = f"{best:.6f} V" if pd.notna(best) else "N/A"
    return f"n={n_total}, ok={n_ok}, failed={n_fail}, best={best_text}"


def plot_objective_evolution(stage1: pd.DataFrame, stage2: pd.DataFrame, combined: pd.DataFrame):
    stage1_end = len(stage1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

    ok = combined[combined["is_ok"]]
    fail = combined[~combined["is_ok"]]

    # -----------------------------
    # 1) Successful evaluations
    # -----------------------------
    ax = axes[0]

    if not ok.empty:
        ax.plot(
            ok["eval_global"],
            ok["objective_V"],
            alpha=0.4,
            label="Objective (successful)",
        )

        ax.plot(
            ok["eval_global"],
            ok["objective_ok_V"].cummin(),
            linewidth=2.5,
            label="Running best",
        )

    ax.axvline(stage1_end + 0.5, linestyle="--", linewidth=1.5, label="Stage boundary")

    ax.set_title("Objective evolution (successful evaluations)")
    ax.set_xlabel("Global evaluation number")
    ax.set_ylabel("RMSE [V]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -----------------------------
    # 2) Running best only
    # -----------------------------
    ax = axes[1]

    if not ok.empty:
        ax.plot(
            ok["eval_global"],
            ok["objective_ok_V"].cummin(),
            linewidth=3,
            label="Running best objective",
        )

    ax.axvline(stage1_end + 0.5, linestyle="--", linewidth=1.5, label="Stage boundary")

    ax.set_title("Running best objective only")
    ax.set_xlabel("Global evaluation number")
    ax.set_ylabel("RMSE [V]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -----------------------------
    # 3) Failed evaluations
    # -----------------------------
    ax = axes[2]

    if not fail.empty:
        ax.scatter(
            fail["eval_global"],
            fail["objective_V"],
            marker="x",
            s=40,
            label="Failed evaluations",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No failed evaluations",
            transform=ax.transAxes,
            ha="center",
        )

    ax.axvline(stage1_end + 0.5, linestyle="--", linewidth=1.5)

    ax.set_title("Failed evaluations")
    ax.set_xlabel("Global evaluation number")
    ax.set_ylabel("RMSE [V]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

def plot_window_rmses(combined: pd.DataFrame):
    rmse_cols = [
        "rmse_ohmic_0_2s_V",
        "rmse_kinetic_2_20s_V",
        "rmse_diffusion_20_120s_V",
        "rmse_capacity_120plus_s_V",
        "rmse_full_profile_V",
    ]

    available = [c for c in rmse_cols if c in combined.columns]
    if not available:
        return None

    ok = combined[combined["is_ok"]].copy()
    if ok.empty:
        return None

    # original intent: one subplot per RMSE metric + one combined running-best subplot
    n_metrics = len(available)
    n_total = n_metrics + 1  # +1 for combined running-best panel

    # fixed 3x2 layout, filled column-wise
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        sharex=True,
        constrained_layout=True,
    )
    axes_flat = axes.flatten(order="F")  # column-major fill

    # metric subplots
    for i, col in enumerate(available):
        ax = axes_flat[i]
        ax.plot(
            ok["eval_global"],
            ok[col],
            linewidth=1.8,
            alpha=0.6,
            label="RMSE",
        )

        running_best = ok[col].cummin()
        ax.plot(
            ok["eval_global"],
            running_best,
            linestyle="--",
            linewidth=2.2,
            label="Running best",
        )

        ax.set_ylabel("RMSE [V]")
        ax.set_title(col.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend()

    # final subplot: combined running-best only
    ax = axes_flat[n_metrics]
    for col in available:
        running_best = ok[col].cummin()
        ax.plot(
            ok["eval_global"],
            running_best,
            linewidth=2,
            label=col,
        )

    ax.set_title("Running best only across RMSE metrics")
    ax.set_ylabel("RMSE [V]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # hide any unused panels if fewer than 5 RMSE columns are present
    for j in range(n_total, len(axes_flat)):
        axes_flat[j].axis("off")

    # x-label only on bottom row visible panels
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Global evaluation number")

    fig.suptitle("RMSE evolution by window (successful evaluations)", fontsize=14)

    return fig

def main() -> None:
    args = parse_args()

    stage1 = load_history(args.stage1, "stage1")
    stage2 = load_history(args.stage2, "stage2")
    combined = build_combined_history(stage1, stage2)

    print("Stage 1:", summarize_stage(stage1))
    print("Stage 2:", summarize_stage(stage2))
    print(
        "Overall best successful objective:",
        f"{combined['objective_ok_V'].min():.6f} V"
        if combined["objective_ok_V"].notna().any()
        else "N/A",
    )

    fig1 = plot_objective_evolution(stage1, stage2, combined)
    fig2 = plot_window_rmses(combined)

    if args.output is not None:
        output_main = args.output
        fig1.savefig(output_main, dpi=200, bbox_inches="tight")
        print(f"Saved main figure to: {output_main}")

        if fig2 is not None:
            output_rmse = output_main.with_name(output_main.stem + "_windows" + output_main.suffix)
            fig2.savefig(output_rmse, dpi=200, bbox_inches="tight")
            print(f"Saved window RMSE figure to: {output_rmse}")
    else:
        plt.show()


if __name__ == "__main__":
    main()