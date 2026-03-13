"""Plot benchmark metrics with smoothing, confidence bands, and chunk summaries.

Example:
    python capstone_agent/plot_benchmarks.py \
      --csv capstone_agent/benchmarks/training_metrics.csv \
      --run-name iter_full \
      --rolling-window 100 \
      --ema-span 100 \
      --chunk-size 100
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required to plot benchmarks. Install with: pip install matplotlib"
    ) from e


def _load(csv_path: str, run_name: str | None, mode: str):
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if run_name:
        df = df[df["run_name"] == run_name]
    if mode != "all":
        df = df[df["mode"] == mode]
    if df.empty:
        raise SystemExit("No rows found for the selected filters.")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def _with_metrics(df: pd.DataFrame, rolling_window: int, ema_span: int, ci_z: float):
    out = df.copy().reset_index(drop=True)
    out["idx"] = np.arange(1, len(out) + 1)
    out["won_f"] = out["won"].astype(float)
    out["reward_f"] = out["reward"].astype(float)

    out["cum_win_rate"] = out["won_f"].expanding().mean()
    out["rolling_win_rate"] = out["won_f"].rolling(rolling_window, min_periods=1).mean()
    out["ema_win_rate"] = out["won_f"].ewm(span=ema_span, adjust=False).mean()

    out["rolling_reward"] = out["reward_f"].rolling(rolling_window, min_periods=1).mean()
    out["ema_reward"] = out["reward_f"].ewm(span=ema_span, adjust=False).mean()

    n = out["won_f"].rolling(rolling_window, min_periods=1).count()
    p = out["rolling_win_rate"]
    se = np.sqrt((p * (1.0 - p)).clip(lower=0.0) / n.clip(lower=1.0))
    out["ci_low"] = (p - ci_z * se).clip(lower=0.0)
    out["ci_high"] = (p + ci_z * se).clip(upper=1.0)
    return out


def _chunk_summary(df: pd.DataFrame, chunk_size: int):
    out = df.copy()
    out["chunk"] = ((out["idx"] - 1) // chunk_size) + 1
    grouped = (
        out.groupby("chunk", as_index=False)
        .agg(
            start_idx=("idx", "min"),
            end_idx=("idx", "max"),
            games=("idx", "count"),
            wins=("won_f", "sum"),
            avg_reward=("reward_f", "mean"),
        )
        .sort_values("chunk")
    )
    grouped["win_rate"] = grouped["wins"] / grouped["games"].clip(lower=1)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Plot Capstone benchmark metrics")
    parser.add_argument(
        "--csv",
        default="capstone_agent/benchmarks/training_metrics.csv",
        help="Path to benchmark CSV written by run_simulation.py",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run_name filter (plots all rows if omitted).",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train", "eval"],
        default="all",
        help="Optional mode filter before plotting.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Rolling window for win-rate/reward smoothing.",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=100,
        help="EMA span for smoother trend lines.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size for checkpoint-style summary plot.",
    )
    parser.add_argument(
        "--ci-z",
        type=float,
        default=1.96,
        help="z-value for confidence interval on rolling win rate.",
    )
    parser.add_argument(
        "--out",
        default="capstone_agent/benchmarks/benchmark_plot.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    df = _load(args.csv, args.run_name, args.mode)
    train_raw = df[df["mode"] == "train"].copy()
    eval_raw = df[df["mode"] == "eval"].copy()

    if train_raw.empty and eval_raw.empty:
        raise SystemExit("No train/eval rows found after filtering.")

    train = (
        _with_metrics(train_raw, args.rolling_window, args.ema_span, args.ci_z)
        if not train_raw.empty
        else None
    )
    eval_df = (
        _with_metrics(eval_raw, args.rolling_window, args.ema_span, args.ci_z)
        if not eval_raw.empty
        else None
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    # --- Win-rate panel ---
    if train is not None:
        axes[0].plot(train["idx"], train["cum_win_rate"], label="Train cumulative")
        axes[0].plot(
            train["idx"],
            train["rolling_win_rate"],
            label=f"Train rolling ({args.rolling_window})",
            alpha=0.85,
        )
        axes[0].plot(
            train["idx"],
            train["ema_win_rate"],
            label=f"Train EMA ({args.ema_span})",
            linewidth=2,
        )
        axes[0].fill_between(
            train["idx"],
            train["ci_low"],
            train["ci_high"],
            alpha=0.15,
            label=f"Train CI (z={args.ci_z:g})",
        )
    if eval_df is not None:
        axes[0].plot(
            eval_df["idx"],
            eval_df["rolling_win_rate"],
            label=f"Eval rolling ({args.rolling_window})",
            linestyle="--",
            linewidth=2,
        )
        axes[0].plot(
            eval_df["idx"],
            eval_df["ema_win_rate"],
            label=f"Eval EMA ({args.ema_span})",
            linestyle=":",
            linewidth=2,
        )

    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # --- Reward panel ---
    if train is not None:
        axes[1].plot(
            train["idx"], train["reward_f"], alpha=0.15, label="Train per-game reward"
        )
        axes[1].plot(
            train["idx"],
            train["rolling_reward"],
            label=f"Train rolling reward ({args.rolling_window})",
            alpha=0.85,
        )
        axes[1].plot(
            train["idx"],
            train["ema_reward"],
            label=f"Train EMA reward ({args.ema_span})",
            linewidth=2,
        )
    if eval_df is not None:
        axes[1].plot(
            eval_df["idx"],
            eval_df["rolling_reward"],
            label=f"Eval rolling reward ({args.rolling_window})",
            linestyle="--",
            linewidth=2,
        )
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # --- Chunk summary panel ---
    if train is not None:
        train_chunk = _chunk_summary(train, args.chunk_size)
        axes[2].plot(
            train_chunk["end_idx"],
            train_chunk["win_rate"],
            marker="o",
            label=f"Train chunk win rate ({args.chunk_size})",
        )
    if eval_df is not None:
        eval_chunk = _chunk_summary(eval_df, args.chunk_size)
        axes[2].plot(
            eval_chunk["end_idx"],
            eval_chunk["win_rate"],
            marker="s",
            linestyle="--",
            label=f"Eval chunk win rate ({args.chunk_size})",
        )
    axes[2].set_ylabel("Chunk win rate")
    axes[2].set_xlabel("Game index")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    title = "Capstone Benchmark Metrics"
    if args.run_name:
        title += f" ({args.run_name})"
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved plot: {args.out}")

    # Brief terminal summary to complement the image in SSH workflows.
    if train is not None and len(train) > 0:
        print(
            "Train summary: "
            f"games={len(train)} "
            f"cum_win={train['cum_win_rate'].iloc[-1]:.3f} "
            f"roll_win={train['rolling_win_rate'].iloc[-1]:.3f} "
            f"ema_win={train['ema_win_rate'].iloc[-1]:.3f}"
        )
    if eval_df is not None and len(eval_df) > 0:
        print(
            "Eval summary: "
            f"games={len(eval_df)} "
            f"cum_win={eval_df['cum_win_rate'].iloc[-1]:.3f} "
            f"roll_win={eval_df['rolling_win_rate'].iloc[-1]:.3f} "
            f"ema_win={eval_df['ema_win_rate'].iloc[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
