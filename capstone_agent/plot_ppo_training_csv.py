"""Plot PPO update metrics from --training-metrics-csv (ppo_training.csv).

Usage:
  python capstone_agent/plot_ppo_training_csv.py \\
    --csv capstone_agent/models/neuner_vs_alphabeta/ppo_training.csv

  python capstone_agent/plot_ppo_training_csv.py --csv path/to/ppo_training.csv \\
    --save capstone_agent/models/neuner_vs_alphabeta/ppo_curves.png

  # smooth noisy curves (rolling mean over N updates)
  python capstone_agent/plot_ppo_training_csv.py --csv ppo_training.csv --rolling 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to ppo_training.csv from run_simulation.py --training-metrics-csv",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, write figure to this path instead of opening a window",
    )
    p.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Rolling mean window over PPO updates (1 = no smoothing)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    for col in (
        "game_index",
        "ppo_total_loss",
        "ppo_actor_loss",
        "ppo_critic_loss",
        "ppo_entropy_mean",
    ):
        if col not in df.columns:
            raise SystemExit(f"CSV missing column {col!r}; got {list(df.columns)}")

    df = df.dropna(subset=["game_index", "ppo_total_loss"])
    df["game_index"] = pd.to_numeric(df["game_index"], errors="coerce")
    for c in ("ppo_total_loss", "ppo_actor_loss", "ppo_critic_loss", "ppo_entropy_mean"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ppo_total_loss"])

    if args.rolling > 1:
        for c in ("ppo_total_loss", "ppo_actor_loss", "ppo_critic_loss", "ppo_entropy_mean"):
            df[c] = df[c].rolling(window=args.rolling, min_periods=1).mean()

    x = df["game_index"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(f"PPO metrics vs game_index ({args.csv.name})", fontsize=12)

    axes[0, 0].plot(x, df["ppo_total_loss"], color="tab:blue", linewidth=0.8)
    axes[0, 0].set_ylabel("ppo_total_loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, df["ppo_actor_loss"], color="tab:orange", linewidth=0.8)
    axes[0, 1].set_ylabel("ppo_actor_loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, df["ppo_critic_loss"], color="tab:green", linewidth=0.8)
    axes[1, 0].set_ylabel("ppo_critic_loss")
    axes[1, 0].set_xlabel("game_index (training game when update ran)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, df["ppo_entropy_mean"], color="tab:red", linewidth=0.8)
    axes[1, 1].set_ylabel("ppo_entropy_mean")
    axes[1, 1].set_xlabel("game_index")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
