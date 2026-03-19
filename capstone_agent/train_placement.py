"""Supervised training for the PlacementAgent.

Loads a .npz dataset produced by collect_placement_data.py, trains
the PlacementModel with weighted cross-entropy, and saves the result.

Usage:
    python capstone_agent/train_placement.py \
        --data capstone_agent/placement_data.npz \
        --out  capstone_agent/models/placement_model.pt \
        --epochs 30
"""

import sys
import os
import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from PlacementModel import PlacementModel
from device import get_device


def load_dataset(path: str):
    d = np.load(path)
    return d["obs"], d["masks"], d["actions"], d["won"]


def train(
    model: PlacementModel,
    obs: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    won: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    win_weight: float = 1.0,
    loss_weight: float = 0.1,
    val_frac: float = 0.1,
    verbose: bool = True,
):
    device = get_device()
    model = model.to(device)

    n = len(obs)
    perm = np.random.permutation(n)
    split = int(n * (1 - val_frac))
    train_idx, val_idx = perm[:split], perm[split:]

    obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
    mask_t = torch.as_tensor(masks, dtype=torch.float32).to(device)
    act_t = torch.as_tensor(actions, dtype=torch.long).to(device)
    weights = torch.where(
        torch.as_tensor(won, dtype=torch.float32) > 0.5,
        win_weight,
        loss_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_perm = torch.as_tensor(
            np.random.permutation(len(train_idx)), dtype=torch.long
        )
        running_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), batch_size):
            bi = train_idx[train_perm[start : start + batch_size]]
            b_obs = obs_t[bi]
            b_mask = mask_t[bi]
            b_act = act_t[bi]
            b_w = weights[bi]

            probs, _ = model(b_obs, b_mask)
            log_probs = torch.log(probs + 1e-8)
            nll = nn.functional.nll_loss(log_probs, b_act, reduction="none")
            loss = (nll * b_w).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            vi = torch.as_tensor(val_idx, dtype=torch.long).to(device)
            v_probs, _ = model(obs_t[vi], mask_t[vi])
            v_log = torch.log(v_probs + 1e-8)
            v_nll = nn.functional.nll_loss(v_log, act_t[vi], reduction="none")
            v_loss = (v_nll * weights[vi]).mean().item()

            preds = v_probs.argmax(dim=-1)
            acc = (preds == act_t[vi]).float().mean().item()

            win_mask = won[val_idx] > 0.5
            if win_mask.sum() > 0:
                wi = torch.as_tensor(val_idx[win_mask], dtype=torch.long).to(device)
                w_probs, _ = model(obs_t[wi], mask_t[wi])
                win_acc = (w_probs.argmax(dim=-1) == act_t[wi]).float().mean().item()
            else:
                win_acc = 0.0

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            train_avg = running_loss / max(n_batches, 1)
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_avg:.4f}  "
                f"val_loss={v_loss:.4f}  "
                f"val_acc={acc:.2%}  "
                f"val_win_acc={win_acc:.2%}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PlacementModel with supervised learning"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="capstone_agent/data/placement_data.npz",
        help="Path to .npz dataset from collect_placement_data.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="capstone_agent/models/placement_model.pt",
        help="Where to save trained weights",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--win-weight", type=float, default=1.0,
        help="Sample weight for actions from winning games",
    )
    parser.add_argument(
        "--loss-weight", type=float, default=0.1,
        help="Sample weight for actions from losing games",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (L2 regularization)",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    obs, masks, actions, won = load_dataset(args.data)
    n_wins = int(won.sum())
    print(
        f"  {len(obs)} samples  ({n_wins} from wins, {len(obs) - n_wins} from losses)"
    )

    device = get_device()
    model = PlacementModel(obs_size=obs.shape[1], hidden_size=args.hidden_size)
    params = sum(p.numel() for p in model.parameters())
    print(f"  PlacementModel: {params:,} params (hidden={args.hidden_size})")
    print(f"  Device: {device}")
    print()

    model = train(
        model,
        obs,
        masks,
        actions,
        won,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        val_frac=args.val_frac,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)

    root, ext = os.path.splitext(args.out)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    ckpt_path = f"{root}_{stamp}{ext}"
    torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest model saved to {args.out}  (checkpoint: {ckpt_path})")


if __name__ == "__main__":
    main()
