"""Online collect-train loop for the PlacementAgent.

Alternates between collecting batches of games and training the
PlacementModel, logging validation metrics after each cycle to a
JSON-lines file.  If pointed at an existing .npz dataset, first
replays the data in increments (catch-up phase) so the learning
curve covers all previously collected samples.

Usage:
    # Start from scratch:
    python capstone_agent/train_placement_online.py \
        --games-per-cycle 5000 --cycles 50 --workers 12

    # Catch up on existing data, then keep collecting:
    python capstone_agent/train_placement_online.py \
        --data capstone_agent/data/placement_data_20260319T1642Z.npz \
        --games-per-cycle 5000 --cycles 50 --workers 12

    # Resume from a model checkpoint (skip catch-up):
    python capstone_agent/train_placement_online.py \
        --data capstone_agent/data/placement_data_20260319T1642Z.npz \
        --model capstone_agent/models/placement_model.pt \
        --games-per-cycle 5000 --cycles 50 --workers 12

    # View learning curve:
    python capstone_agent/plot_training.py
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PlacementModel import PlacementModel
from device import get_device
from train_placement import train as train_model, load_dataset
from collect_placement_data import collect


def _log_entry(log_path, entry):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_cycle(entry):
    phase = entry.get("phase", "online")
    tag = "catch-up" if phase == "catchup" else f"cycle {entry['cycle']}"
    print(
        f"  [{tag}] "
        f"samples={entry['total_samples']:,}  "
        f"val_loss={entry['val_loss']:.4f}  "
        f"val_acc={entry['val_acc']:.2%}  "
        f"val_win_acc={entry['val_win_acc']:.2%}  "
        f"elapsed={entry['elapsed_s']:.0f}s",
        flush=True,
    )


def _train_on_slice(obs, masks, actions, won, args, device, out_model):
    """Create a fresh model, train on the given data, return (model, metrics)."""
    model = PlacementModel(
        obs_size=obs.shape[1], hidden_size=args.hidden_size
    ).to(device)

    model, metrics = train_model(
        model, obs, masks, actions, won,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        val_frac=args.val_frac,
        out_path=out_model,
        verbose=False,
    )
    return model, metrics


def run(args):
    device = get_device()
    t0 = time.time()

    data_path = args.data
    out_model = args.model
    log_path = args.log
    samples_per_step = args.games_per_cycle * 4

    print(f"Online training loop")
    print(f"  Device: {device}")
    print(f"  Data:   {data_path}")
    print(f"  Model:  {out_model}")
    print(f"  Log:    {log_path}")
    print(f"  Games/cycle: {args.games_per_cycle}  "
          f"Epochs/cycle: {args.epochs}  "
          f"Workers: {args.workers or 'auto'}")
    print()

    # ------------------------------------------------------------------
    # Catch-up phase: replay existing data in increments
    # ------------------------------------------------------------------
    existing_samples = 0
    skip_catchup = not args.catch_up

    if os.path.exists(data_path):
        obs, masks, actions, won = load_dataset(data_path)
        existing_samples = len(obs)
        print(f"  Existing data: {existing_samples:,} samples")

        if not skip_catchup and existing_samples > 0:
            print(f"\n--- Catch-up phase: training on existing data "
                  f"in steps of ~{samples_per_step:,} samples ---\n")

            cycle_num = 0
            for end in range(samples_per_step, existing_samples + 1,
                             samples_per_step):
                end = min(end, existing_samples)
                print(f"  Training on data[:{end:,}] ...")
                model, metrics = _train_on_slice(
                    obs[:end], masks[:end], actions[:end], won[:end],
                    args, device, out_model,
                )
                entry = {
                    "cycle": cycle_num,
                    "phase": "catchup",
                    "total_samples": int(end),
                    "val_loss": metrics["val_loss"],
                    "val_acc": metrics["val_acc"],
                    "val_win_acc": metrics["val_win_acc"],
                    "best_val_loss": metrics["best_val_loss"],
                    "elapsed_s": round(time.time() - t0, 1),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                _log_entry(log_path, entry)
                _print_cycle(entry)
                cycle_num += 1

            if existing_samples % samples_per_step != 0:
                print(f"  Training on data[:{existing_samples:,}] (remainder) ...")
                model, metrics = _train_on_slice(
                    obs, masks, actions, won,
                    args, device, out_model,
                )
                entry = {
                    "cycle": cycle_num,
                    "phase": "catchup",
                    "total_samples": int(existing_samples),
                    "val_loss": metrics["val_loss"],
                    "val_acc": metrics["val_acc"],
                    "val_win_acc": metrics["val_win_acc"],
                    "best_val_loss": metrics["best_val_loss"],
                    "elapsed_s": round(time.time() - t0, 1),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                _log_entry(log_path, entry)
                _print_cycle(entry)

            print(f"\n  Catch-up complete.\n")
    else:
        print(f"  No existing data at {data_path} — starting fresh\n")

    # ------------------------------------------------------------------
    # Online loop: collect → train → log → repeat
    # ------------------------------------------------------------------
    print(f"--- Online loop: {args.cycles} cycles of "
          f"{args.games_per_cycle} games each ---\n")

    try:
        for c in range(1, args.cycles + 1):
            print(f"  Cycle {c}/{args.cycles}: collecting "
                  f"{args.games_per_cycle} games ...")
            collect(
                num_games=args.games_per_cycle,
                blue_type=args.blue,
                enemy_type=args.enemy,
                out_path=data_path,
                append_path=data_path if os.path.exists(data_path) else None,
                num_workers=args.workers,
            )

            print(f"  Cycle {c}/{args.cycles}: training ...")
            obs, masks, actions, won = load_dataset(data_path)

            model = PlacementModel(
                obs_size=obs.shape[1], hidden_size=args.hidden_size
            ).to(device)
            if os.path.exists(out_model):
                model.load_state_dict(
                    torch.load(out_model, map_location=device,
                               weights_only=True)
                )

            model, metrics = train_model(
                model, obs, masks, actions, won,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                win_weight=args.win_weight,
                loss_weight=args.loss_weight,
                val_frac=args.val_frac,
                out_path=out_model,
                verbose=False,
            )

            entry = {
                "cycle": c,
                "phase": "online",
                "total_samples": int(len(obs)),
                "val_loss": metrics["val_loss"],
                "val_acc": metrics["val_acc"],
                "val_win_acc": metrics["val_win_acc"],
                "best_val_loss": metrics["best_val_loss"],
                "elapsed_s": round(time.time() - t0, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _log_entry(log_path, entry)
            _print_cycle(entry)

    except KeyboardInterrupt:
        print(f"\n  Stopped after cycle {c - 1}.", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed / 60:.1f} min")
    print(f"  Log: {log_path}")
    print(f"  Plot: python capstone_agent/plot_training.py --log {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Online collect-train loop for PlacementAgent"
    )

    parser.add_argument(
        "--data", type=str,
        default="capstone_agent/data/placement_data.npz",
        help="Path to .npz data file (created if missing, appended each cycle)",
    )
    parser.add_argument(
        "--model", type=str,
        default="capstone_agent/models/placement_model.pt",
        help="Path to model checkpoint (created/overwritten each cycle)",
    )
    parser.add_argument(
        "--log", type=str,
        default="capstone_agent/data/training_log.jsonl",
        help="Path to JSON-lines training log",
    )

    parser.add_argument(
        "--catch-up", action=argparse.BooleanOptionalAction, default=True,
        help="Train on existing data in increments before online loop "
             "(use --no-catch-up to skip)",
    )
    parser.add_argument("--games-per-cycle", type=int, default=5000)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--win-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=64)

    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--blue", type=str, default="alphabeta",
        help="Bot type for Blue (the player we learn from)",
    )
    parser.add_argument(
        "--enemy", type=str, default="alphabeta",
        help="Bot type for Red (the opponent)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
