"""Plot training/evaluation benchmark metrics from run_simulation CSV logs.

Example:
    python capstone_agent/plot_benchmarks.py \
      --csv capstone_agent/benchmarks/training_metrics.csv \
      --run-name run_20260223T120000Z
"""

import argparse
import os

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required to plot benchmarks. Install with: pip install matplotlib"
    ) from e


def _load(csv_path: str, run_name: str | None):
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if run_name:
        df = df[df["run_name"] == run_name]
    if df.empty:
        raise SystemExit("No rows found for the selected filters.")
    df = df.sort_values("timestamp_utc")
    return df


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
        "--rolling-window",
        type=int,
        default=20,
        help="Rolling window for win-rate/reward smoothing.",
    )
    parser.add_argument(
        "--out",
        default="capstone_agent/benchmarks/benchmark_plot.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    df = _load(args.csv, args.run_name)
    df = df.reset_index(drop=True)
    x = range(1, len(df) + 1)

    won = df["won"].astype(float)
    reward = df["reward"].astype(float)
    rolling_wins = won.rolling(args.rolling_window, min_periods=1).mean()
    rolling_reward = reward.rolling(args.rolling_window, min_periods=1).mean()
    cumulative_win_rate = won.expanding().mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, cumulative_win_rate, label="Cumulative win rate")
    axes[0].plot(x, rolling_wins, label=f"Rolling win rate ({args.rolling_window})")
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, reward, alpha=0.25, label="Per-game reward")
    axes[1].plot(
        x, rolling_reward, linewidth=2, label=f"Rolling reward ({args.rolling_window})"
    )
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Game index")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    title = "Capstone Benchmark Metrics"
    if args.run_name:
        title += f" ({args.run_name})"
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
