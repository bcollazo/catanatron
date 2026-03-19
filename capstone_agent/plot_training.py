"""Plot learning curves from the online training log.

Reads the JSON-lines file produced by train_placement_online.py and
shows val_loss, val_acc, and val_win_acc vs total training samples.

Usage:
    python capstone_agent/plot_training.py
    python capstone_agent/plot_training.py --log capstone_agent/data/training_log.jsonl
    python capstone_agent/plot_training.py --save capstone_agent/data/learning_curve.png
"""

import json
import argparse

import matplotlib.pyplot as plt


def load_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot(entries, save_path=None):
    samples = [e["total_samples"] for e in entries]
    val_loss = [e["val_loss"] for e in entries]
    val_acc = [e["val_acc"] * 100 for e in entries]
    val_win_acc = [e["val_win_acc"] * 100 for e in entries]

    catchup_mask = [e.get("phase") == "catchup" for e in entries]
    online_mask = [not c for c in catchup_mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Placement Model — Learning Curve", fontsize=14)

    # --- Top panel: val_loss ---
    ax1.plot(samples, val_loss, "o-", markersize=4, color="tab:red",
             label="val_loss")
    if any(catchup_mask):
        s = [samples[i] for i, m in enumerate(catchup_mask) if m]
        v = [val_loss[i] for i, m in enumerate(catchup_mask) if m]
        ax1.fill_between(s, 0, max(val_loss) * 1.05, alpha=0.07,
                         color="gray", label="catch-up phase")
    ax1.set_ylabel("Validation Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: val_acc + val_win_acc ---
    ax2.plot(samples, val_acc, "o-", markersize=4, color="tab:blue",
             label="val_acc")
    ax2.plot(samples, val_win_acc, "s-", markersize=4, color="tab:green",
             label="val_win_acc")
    if any(catchup_mask):
        s = [samples[i] for i, m in enumerate(catchup_mask) if m]
        ax2.fill_between(s, 0, 100, alpha=0.07, color="gray",
                         label="catch-up phase")
    ax2.set_xlabel("Total Training Samples")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves from online training log"
    )
    parser.add_argument(
        "--log", type=str,
        default="capstone_agent/data/training_log.jsonl",
        help="Path to JSON-lines training log",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save plot to file instead of showing interactively",
    )
    args = parser.parse_args()

    entries = load_log(args.log)
    if not entries:
        print(f"No entries found in {args.log}")
        return

    print(f"Loaded {len(entries)} log entries from {args.log}")
    print(f"  Samples range: {entries[0]['total_samples']:,} — "
          f"{entries[-1]['total_samples']:,}")

    plot(entries, save_path=args.save)


if __name__ == "__main__":
    main()
