"""Plot benchmark metrics with smoothing, confidence bands, and chunk summaries.

Trailing win rates default to 500, 5000, and 50000 games (strict windows). Use
--rolling-windows to customize. Per-game reward smoothing still uses --rolling-window.

Example:
    python capstone_agent/plot_benchmarks.py \
      --csv capstone_agent/benchmarks/training_metrics.csv \
      --run-name iter_full \
      --rolling-windows 500,5000,50000 \
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


def _parse_rolling_windows(spec: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        w = int(part)
        if w < 1:
            raise SystemExit(f"rolling window must be >= 1, got {w}")
        if w not in seen:
            seen.add(w)
            out.append(w)
    if not out:
        return [500, 5000, 50000]
    return out


def _with_metrics(
    df: pd.DataFrame,
    reward_rolling_window: int,
    ema_span: int,
    ci_z: float,
    rolling_win_windows: list[int],
):
    out = df.copy().reset_index(drop=True)
    out["idx"] = np.arange(1, len(out) + 1)
    out["won_f"] = out["won"].astype(float)
    out["reward_f"] = out["reward"].astype(float)

    out["cum_win_rate"] = out["won_f"].expanding().mean()
    for w in rolling_win_windows:
        col = f"rolling_win_rate_{w}"
        out[col] = out["won_f"].rolling(w, min_periods=w).mean()

    ci_window = min(rolling_win_windows)
    out["rolling_win_rate"] = out[f"rolling_win_rate_{ci_window}"].copy()
    out["ema_win_rate"] = out["won_f"].ewm(span=ema_span, adjust=False).mean()

    out["rolling_reward"] = out["reward_f"].rolling(
        reward_rolling_window, min_periods=1
    ).mean()
    out["ema_reward"] = out["reward_f"].ewm(span=ema_span, adjust=False).mean()

    n = out["won_f"].rolling(ci_window, min_periods=ci_window).count()
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
        help="Rolling window for per-game reward smoothing only.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default="500,5000,50000",
        help=(
            "Comma-separated trailing win-rate windows (strict: NaN until N games). "
            "CI uses the smallest window."
        ),
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

    win_windows = _parse_rolling_windows(args.rolling_windows)
    ci_w = min(win_windows)

    train = (
        _with_metrics(
            train_raw, args.rolling_window, args.ema_span, args.ci_z, win_windows
        )
        if not train_raw.empty
        else None
    )
    eval_df = (
        _with_metrics(
            eval_raw, args.rolling_window, args.ema_span, args.ci_z, win_windows
        )
        if not eval_raw.empty
        else None
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    # --- Win-rate panel ---
    if train is not None:
        axes[0].plot(
            train["idx"], train["cum_win_rate"], label="Train cumulative", color="0.35"
        )
        for w in win_windows:
            col = f"rolling_win_rate_{w}"
            axes[0].plot(
                train["idx"],
                train[col],
                label=f"Train trailing {w} games",
                linewidth=1.4 if w <= 1000 else 1.8,
                alpha=0.9 if w <= 1000 else 1.0,
            )
        axes[0].plot(
            train["idx"],
            train["ema_win_rate"],
            label=f"Train EMA ({args.ema_span})",
            linewidth=1.5,
            linestyle=":",
            color="0.25",
        )
        axes[0].fill_between(
            train["idx"],
            train["ci_low"],
            train["ci_high"],
            alpha=0.12,
            label=f"Train CI on {ci_w}-game window (z={args.ci_z:g})",
        )
    if eval_df is not None:
        for w in win_windows:
            col = f"rolling_win_rate_{w}"
            axes[0].plot(
                eval_df["idx"],
                eval_df[col],
                label=f"Eval trailing {w} games",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
        axes[0].plot(
            eval_df["idx"],
            eval_df["ema_win_rate"],
            label=f"Eval EMA ({args.ema_span})",
            linestyle=":",
            linewidth=1.5,
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
        parts = [
            f"games={len(train)}",
            f"cum_win={train['cum_win_rate'].iloc[-1]:.3f}",
            f"ema_win={train['ema_win_rate'].iloc[-1]:.3f}",
        ]
        for w in win_windows:
            col = f"rolling_win_rate_{w}"
            v = train[col].iloc[-1]
            parts.append(f"trail{w}={v:.3f}" if pd.notna(v) else f"trail{w}=nan")
        print("Train summary: " + " ".join(parts))
    if eval_df is not None and len(eval_df) > 0:
        parts = [
            f"games={len(eval_df)}",
            f"cum_win={eval_df['cum_win_rate'].iloc[-1]:.3f}",
            f"ema_win={eval_df['ema_win_rate'].iloc[-1]:.3f}",
        ]
        for w in win_windows:
            col = f"rolling_win_rate_{w}"
            v = eval_df[col].iloc[-1]
            parts.append(f"trail{w}={v:.3f}" if pd.notna(v) else f"trail{w}=nan")
        print("Eval summary: " + " ".join(parts))


if __name__ == "__main__":
    main()
