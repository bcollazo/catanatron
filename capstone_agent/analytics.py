"""Unified analytics CLI for Capstone training and game JSON datasets.

Subcommands:
    benchmarks      Plot benchmark metrics: cumulative win rate, trailing win rates
                      (default 500 / 5k / 50k games), EMA, reward rollings, chunk summary.
    first-second    Plot wins when going first vs second in fixed chunks.
    placements      Aggregate initial placement stats from replay JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _require_pandas():
    try:
        import pandas as pd
    except Exception as e:
        raise SystemExit("pandas is required. Install with: pip install pandas") from e
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting analytics. "
            "Install with: pip install matplotlib"
        ) from e
    return plt


def _load_benchmark_csv(csv_path: str, run_name: str | None, mode: str) -> pd.DataFrame:
    pd = _require_pandas()
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
    """Parse '500,5000,50000' into ordered unique positive window sizes."""
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
) -> pd.DataFrame:
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


def _chunk_summary(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
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


def cmd_benchmarks(args):
    plt = _require_matplotlib()
    df = _load_benchmark_csv(args.csv, args.run_name, args.mode)
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

    pd = _require_pandas()
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


def _load_first_second(csv_path: str, run_name: str | None, mode: str) -> pd.DataFrame:
    df = _load_benchmark_csv(csv_path, run_name, mode)
    if "went_first" not in df.columns:
        raise SystemExit(
            "CSV does not contain 'went_first'. This metric is available in new "
            "rows after the seat-logging update in run_simulation.py."
        )
    df = df.dropna(subset=["went_first"]).copy()
    if df.empty:
        raise SystemExit("No rows with went_first values found after filtering.")
    df["went_first"] = df["went_first"].astype(int)
    df["won"] = df["won"].astype(int)
    if "game_index" in df.columns:
        df = df.sort_values("game_index")
    df = df.reset_index(drop=True)
    df["row_idx"] = np.arange(1, len(df) + 1)
    return df


def _chunk_first_second(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    c = df.copy()
    c["chunk"] = ((c["row_idx"] - 1) // chunk_size) + 1
    grouped = []
    for _, g in c.groupby("chunk"):
        first = g[g["went_first"] == 1]
        second = g[g["went_first"] == 0]
        first_games = len(first)
        second_games = len(second)
        first_wins = int(first["won"].sum())
        second_wins = int(second["won"].sum())
        grouped.append(
            {
                "start_idx": int(g["row_idx"].min()),
                "end_idx": int(g["row_idx"].max()),
                "games": int(len(g)),
                "first_games": first_games,
                "second_games": second_games,
                "first_wins": first_wins,
                "second_wins": second_wins,
                "first_win_rate": (
                    first_wins / first_games if first_games > 0 else np.nan
                ),
                "second_win_rate": (
                    second_wins / second_games if second_games > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(grouped)


def cmd_first_second(args):
    plt = _require_matplotlib()
    df = _load_first_second(args.csv, args.run_name, args.mode)
    chunks = _chunk_first_second(df, args.chunk_size)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    x = chunks["end_idx"]
    width = max(1, args.chunk_size // 3)

    axes[0].bar(
        x - width / 2,
        chunks["first_wins"],
        width=width,
        label="Wins when we went first",
    )
    axes[0].bar(
        x + width / 2,
        chunks["second_wins"],
        width=width,
        label="Wins when we went second",
    )
    axes[0].set_ylabel("Wins per chunk")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        x,
        chunks["first_win_rate"],
        marker="o",
        label="Win rate when first",
    )
    axes[1].plot(
        x,
        chunks["second_win_rate"],
        marker="s",
        label="Win rate when second",
    )
    axes[1].set_xlabel("Game index (chunk end)")
    axes[1].set_ylabel("Win rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    title = f"First vs Second Performance ({args.chunk_size}-Game Chunks)"
    if args.run_name:
        title += f" - {args.run_name}"
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved plot: {args.out}")

    print("\nChunk summary:")
    for _, row in chunks.iterrows():
        print(
            f"games {int(row['start_idx']):4d}-{int(row['end_idx']):4d} | "
            f"first W:{int(row['first_wins']):3d}/{int(row['first_games']):3d} "
            f"({row['first_win_rate']:.3f}) | "
            f"second W:{int(row['second_wins']):3d}/{int(row['second_games']):3d} "
            f"({row['second_win_rate']:.3f})"
        )


def _get_initial_settlements(data: dict) -> list[tuple[str, int]]:
    colors = data.get("colors", [])
    num_players = len(colors)
    if num_players < 2:
        return []
    expected_settlements = 2 * num_players
    settlements: list[tuple[str, int]] = []
    for record in data.get("action_records", []):
        action = record[0]
        if not action or action[1] != "BUILD_SETTLEMENT":
            continue
        color, _, node_id = action
        settlements.append((color, node_id))
        if len(settlements) >= expected_settlements:
            break
    return settlements


def _resources_for_node(data: dict, node_id: int) -> list[tuple[str, int]]:
    adj = data.get("adjacent_tiles", {})
    key = str(node_id)
    if key not in adj:
        return []
    result = []
    for tile in adj[key]:
        if tile.get("type") == "DESERT":
            continue
        res = tile.get("resource")
        num = tile.get("number")
        if res and num is not None:
            result.append((res, num))
    return result


def cmd_placements(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    resource_counts_winner: dict[str, int] = defaultdict(int)
    resource_counts_loser: dict[str, int] = defaultdict(int)
    number_counts_winner: dict[int, int] = defaultdict(int)
    number_counts_loser: dict[int, int] = defaultdict(int)
    pair_counts_winner: dict[tuple[str, int], int] = defaultdict(int)
    pair_counts_loser: dict[tuple[str, int], int] = defaultdict(int)
    games_processed = 0
    skip_reasons: dict[str, list[str]] = defaultdict(list)

    for path in sorted(data_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            skip_reasons["load_error"].append(path.name)
            continue

        winning_color = data.get("winning_color")
        if not winning_color:
            skip_reasons["no_winning_color"].append(path.name)
            continue

        colors = data.get("colors", [])
        expected_settlements = 2 * len(colors) if len(colors) >= 2 else 8
        settlements = _get_initial_settlements(data)
        if len(settlements) < expected_settlements:
            skip_reasons["incomplete_initial_placements"].append(path.name)
            continue

        for color, node_id in settlements:
            pairs = _resources_for_node(data, node_id)
            is_winner = color == winning_color
            for res, num in pairs:
                if is_winner:
                    resource_counts_winner[res] += 1
                    number_counts_winner[num] += 1
                    pair_counts_winner[(res, num)] += 1
                else:
                    resource_counts_loser[res] += 1
                    number_counts_loser[num] += 1
                    pair_counts_loser[(res, num)] += 1
        games_processed += 1

    total_winner_tiles = sum(resource_counts_winner.values())
    total_loser_tiles = sum(resource_counts_loser.values())
    games_skipped = sum(len(v) for v in skip_reasons.values())

    def pct(c, total):
        return 100 * c / total if total else 0

    print("=" * 60)
    print("INITIAL PLACEMENT STATS")
    print("=" * 60)
    print(f"Games analyzed: {games_processed}  (skipped: {games_skipped})")
    if skip_reasons:
        print("\nSkip reasons:")
        for reason, files in sorted(skip_reasons.items(), key=lambda x: -len(x[1])):
            print(f"  {reason}: {len(files)} games")
    print(
        f"Winner tile touches: {total_winner_tiles}  |  "
        f"Loser tile touches: {total_loser_tiles}"
    )
    print()

    print("--- RESOURCES (winners vs losers) ---")
    print(f"{'Resource':<10} {'Winners':>12} {'%':>8}  {'Losers':>12} {'%':>8}")
    print("-" * 52)
    for res in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]:
        w = resource_counts_winner[res]
        l = resource_counts_loser[res]
        print(
            f"{res:<10} {w:>12} {pct(w, total_winner_tiles):>7.1f}%  "
            f"{l:>12} {pct(l, total_loser_tiles):>7.1f}%"
        )
    print()

    print("--- TOP (resource, number) pairs - WINNERS ---")
    for (res, num), c in sorted(pair_counts_winner.items(), key=lambda x: -x[1])[:15]:
        print(f"  {res} on {num}: {c} ({pct(c, total_winner_tiles):.1f}%)")
    print("\n--- TOP (resource, number) pairs - LOSERS ---")
    for (res, num), c in sorted(pair_counts_loser.items(), key=lambda x: -x[1])[:15]:
        print(f"  {res} on {num}: {c} ({pct(c, total_loser_tiles):.1f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified analytics CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser(
        "benchmarks", help="Plot benchmark metrics (old plot_benchmarks.py)"
    )
    p_bench.add_argument(
        "--csv",
        default="capstone_agent/benchmarks/training_metrics.csv",
        help="Path to benchmark CSV written by run_simulation.py",
    )
    p_bench.add_argument("--run-name", default=None, help="Optional run_name filter.")
    p_bench.add_argument(
        "--mode",
        choices=["all", "train", "eval"],
        default="all",
        help="Optional mode filter before plotting.",
    )
    p_bench.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Rolling window for per-game reward smoothing (win rate uses --rolling-windows).",
    )
    p_bench.add_argument(
        "--rolling-windows",
        type=str,
        default="500,5000,50000",
        help=(
            "Comma-separated trailing win-rate windows. Each series uses the last N "
            "games only (NaN until N games exist). CI band uses the smallest window."
        ),
    )
    p_bench.add_argument("--ema-span", type=int, default=100)
    p_bench.add_argument("--chunk-size", type=int, default=100)
    p_bench.add_argument("--ci-z", type=float, default=1.96)
    p_bench.add_argument(
        "--out",
        default="capstone_agent/benchmarks/benchmark_plot.png",
        help="Output image path.",
    )
    p_bench.set_defaults(func=cmd_benchmarks)

    p_fs = sub.add_parser(
        "first-second", help="Plot wins when we went first vs second."
    )
    p_fs.add_argument(
        "--csv",
        default="capstone_agent/benchmarks/training_metrics.csv",
        help="Path to benchmark CSV written by run_simulation.py",
    )
    p_fs.add_argument("--run-name", default=None, help="Optional run_name filter.")
    p_fs.add_argument(
        "--mode",
        choices=["all", "train", "eval"],
        default="train",
        help="Optional mode filter before plotting.",
    )
    p_fs.add_argument("--chunk-size", type=int, default=200)
    p_fs.add_argument(
        "--out",
        default="capstone_agent/benchmarks/first_vs_second_200.png",
        help="Output image path.",
    )
    p_fs.set_defaults(func=cmd_first_second)

    p_place = sub.add_parser(
        "placements",
        help="Analyze initial placement resource/number patterns from game JSONs.",
    )
    p_place.add_argument(
        "--data-dir",
        default="my-data-path",
        help="Directory of replay JSON files.",
    )
    p_place.set_defaults(func=cmd_placements)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
