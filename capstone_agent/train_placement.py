"""Legacy flat-dataset supervised trainer for the compact placement model.

This script keeps the original `collect_placement_data.py -> train_placement.py`
workflow working after the placement policy moved to compact observations and
prompt-specific local actions. It accepts the old per-decision flat dataset:

- `obs`     `(N, 1259)` full Capstone observations, or compact observations
- `masks`   `(N, 245)` full Capstone legality masks
- `actions` `(N,)` chosen Capstone placement action indices
- `won`     `(N,)` 1.0 if the acting player won that game

For the new chunked compact supervised pipeline, use
`train_compact_placement_supervised.py` instead.
"""

import sys
import os
import argparse
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from PlacementAgent import PlacementAgent
from device import get_device


def load_dataset(path: str):
    with np.load(path) as loaded:
        if "schema_version" in loaded and "obs" not in loaded:
            raise ValueError(
                "This file looks like a compact chunked placement dataset. "
                "Use `capstone_agent/train_compact_placement_supervised.py` instead."
            )
        return loaded["obs"], loaded["masks"], loaded["actions"], loaded["won"]


def _save_model(model, out_path):
    """Save model weights to canonical path + timestamped checkpoint.

    Writes to a temp file first, then atomically renames to prevent
    corruption if the process is killed mid-write.
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    tmp = out_path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, out_path)

    root, ext = os.path.splitext(out_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    ckpt_path = f"{root}_{stamp}{ext}"
    tmp_ckpt = ckpt_path + ".tmp"
    torch.save(model.state_dict(), tmp_ckpt)
    os.replace(tmp_ckpt, ckpt_path)

    return ckpt_path


def _predict_local_actions(agent: PlacementAgent, obs_t, prompt_t, mask_t):
    settlement_logits, road_logits, _ = agent.model(obs_t)
    preds = torch.empty(len(obs_t), dtype=torch.long, device=obs_t.device)

    settlement_rows = prompt_t == 0
    if settlement_rows.any():
        settlement_mask = mask_t[settlement_rows, :settlement_logits.shape[1]]
        masked_logits = settlement_logits[settlement_rows].masked_fill(
            settlement_mask <= 0,
            -1e9,
        )
        preds[settlement_rows] = masked_logits.argmax(dim=-1)

    road_rows = prompt_t == 1
    if road_rows.any():
        road_mask = mask_t[road_rows, :road_logits.shape[1]]
        masked_logits = road_logits[road_rows].masked_fill(road_mask <= 0, -1e9)
        preds[road_rows] = masked_logits.argmax(dim=-1)

    return preds


def _evaluate_prepared(agent, obs_t, prompt_t, mask_t, act_t, weights, won_np):
    if len(obs_t) == 0:
        return {
            "loss": float("inf"),
            "acc": 0.0,
            "win_acc": 0.0,
        }

    agent.model.eval()
    with torch.no_grad():
        log_probs, _, _ = agent._evaluate_actions(obs_t, prompt_t, mask_t, act_t)
        nll = -log_probs
        loss = (nll * weights).mean().item()

        preds = _predict_local_actions(agent, obs_t, prompt_t, mask_t)
        acc = (preds == act_t).float().mean().item()

        win_mask_np = won_np > 0.5
        if win_mask_np.sum() > 0:
            win_mask_t = torch.as_tensor(win_mask_np, dtype=torch.bool, device=obs_t.device)
            win_acc = (
                (preds[win_mask_t] == act_t[win_mask_t]).float().mean().item()
            )
        else:
            win_acc = 0.0

    return {
        "loss": loss,
        "acc": acc,
        "win_acc": win_acc,
    }


def train(
    agent: PlacementAgent,
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
    out_path: str = None,
    verbose: bool = True,
):
    device = get_device()
    agent.device = device
    agent.model = agent.model.to(device)

    compact_obs = agent._prepare_supervised_obs(obs)
    prompts, local_masks, local_actions = agent._prepare_supervised_actions_and_masks(
        masks, actions
    )

    n = len(compact_obs)
    perm = np.random.permutation(n)
    split = int(n * (1 - val_frac))
    train_idx, val_idx = perm[:split], perm[split:]

    obs_t = torch.as_tensor(compact_obs, dtype=torch.float32, device=device)
    prompt_t = torch.as_tensor(prompts, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(local_masks, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(local_actions, dtype=torch.long, device=device)
    weights = torch.where(
        torch.as_tensor(won, dtype=torch.float32) > 0.5,
        win_weight,
        loss_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        agent.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    metrics = {"val_loss": float("inf"), "val_acc": 0.0, "val_win_acc": 0.0}

    try:
        for epoch in range(1, epochs + 1):
            agent.model.train()
            train_perm = torch.as_tensor(
                np.random.permutation(len(train_idx)), dtype=torch.long
            )
            running_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_idx), batch_size):
                bi = train_idx[train_perm[start : start + batch_size]]
                b_obs = obs_t[bi]
                b_prompt = prompt_t[bi]
                b_mask = mask_t[bi]
                b_act = act_t[bi]
                b_w = weights[bi]

                log_probs, _, _ = agent._evaluate_actions(
                    b_obs,
                    b_prompt,
                    b_mask,
                    b_act,
                )
                nll = -log_probs
                loss = (nll * b_w).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            scheduler.step()

            vi = torch.as_tensor(val_idx, dtype=torch.long, device=device)
            val_metrics = _evaluate_prepared(
                agent,
                obs_t[vi],
                prompt_t[vi],
                mask_t[vi],
                act_t[vi],
                weights[vi],
                won[val_idx],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in agent.model.state_dict().items()
                }
            metrics = {
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_win_acc": val_metrics["win_acc"],
            }

            if verbose:
                train_avg = running_loss / max(n_batches, 1)
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_avg:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_acc={val_metrics['acc']:.2%}  "
                    f"val_win_acc={val_metrics['win_acc']:.2%}"
                )

    except KeyboardInterrupt:
        print(f"\n  Interrupted at epoch {epoch}.", flush=True)

    if best_state is not None:
        agent.model.load_state_dict(best_state)

    if out_path:
        ckpt = _save_model(agent.model, out_path)
        print(f"\nBest model saved to {out_path}  (checkpoint: {ckpt})")
        print(f"  Resume with: --resume {out_path}")

    metrics = {
        **metrics,
        "best_val_loss": best_val_loss,
    }
    return agent, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Legacy flat-dataset trainer for the compact PlacementModel"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="capstone_agent/data/placement_data.npz",
        help="Path to flat .npz dataset from collect_placement_data.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="capstone_agent/models/placement_model.pt",
        help="Where to save trained weights",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing .pt checkpoint to resume training from",
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
    agent = PlacementAgent(hidden_size=args.hidden_size)

    if args.resume:
        agent.load(args.resume)
        print(f"  Resumed from {args.resume}")

    params = sum(p.numel() for p in agent.model.parameters())
    print(f"  PlacementModel: {params:,} params (hidden={args.hidden_size})")
    print(f"  Device: {device}")
    print()

    train(
        agent,
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
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
