"""Train the compact placement model from chunked supervised data."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(__file__))

from PlacementModel import PlacementModel
from device import get_device
from placement_action_space import PlacementPrompt
from placement_features import COMPACT_PLACEMENT_FEATURE_SIZE
from placement_supervised_dataset import (
    load_chunk_records,
    iter_reconstructed_examples,
)


def _save_model(model, out_path):
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


def _append_metrics(metrics_path: str, row: dict):
    if metrics_path is None:
        return
    metrics_dir = os.path.dirname(metrics_path) or "."
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resolve_chunk_paths(inputs: list[str]) -> list[str]:
    paths = []
    for value in inputs:
        if os.path.isdir(value):
            paths.extend(sorted(glob.glob(os.path.join(value, "*.npz"))))
        else:
            paths.append(value)
    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    if not unique_paths:
        raise ValueError("No compact placement chunk files were found")
    return unique_paths


def _load_examples(chunk_paths, selection_mode, win_weight, loss_weight):
    xs, prompts, masks, targets, weights, winner_flags = [], [], [], [], [], []
    games_loaded = 0

    for chunk_path in chunk_paths:
        for record in load_chunk_records(chunk_path):
            games_loaded += 1
            for example in iter_reconstructed_examples(
                record,
                selection_mode=selection_mode,
                win_weight=win_weight,
                loss_weight=loss_weight,
            ):
                xs.append(example.x)
                prompts.append(int(example.prompt))
                padded_mask = np.zeros(PlacementModel.EDGE_ACTION_SIZE, dtype=np.float32)
                padded_mask[: len(example.mask)] = example.mask
                masks.append(padded_mask)
                targets.append(example.target)
                weights.append(example.weight)
                winner_flags.append(example.actor_seat == example.winner_seat)

    if not xs:
        raise ValueError("No supervised examples were reconstructed from the input chunks")

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(prompts, dtype=np.int64),
        np.asarray(masks, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np.asarray(winner_flags, dtype=np.bool_),
        games_loaded,
    )


def _log_probs_for_batch(model, obs_t, prompt_t, mask_t, act_t):
    settlement_logits, road_logits, _ = model(obs_t)
    log_probs = torch.empty(len(obs_t), dtype=torch.float32, device=obs_t.device)
    entropies = torch.empty(len(obs_t), dtype=torch.float32, device=obs_t.device)

    for prompt, logits in (
        (PlacementPrompt.SETTLEMENT, settlement_logits),
        (PlacementPrompt.ROAD, road_logits),
    ):
        rows = prompt_t == int(prompt)
        if not rows.any():
            continue
        prompt_mask = mask_t[rows, : logits.shape[1]]
        masked_logits = logits[rows].masked_fill(prompt_mask <= 0, -1e9)
        dist = Categorical(logits=masked_logits)
        log_probs[rows] = dist.log_prob(act_t[rows])
        entropies[rows] = dist.entropy()

    return log_probs, entropies


def _predict_local_actions(model, obs_t, prompt_t, mask_t):
    settlement_logits, road_logits, _ = model(obs_t)
    preds = torch.empty(len(obs_t), dtype=torch.long, device=obs_t.device)

    settlement_rows = prompt_t == int(PlacementPrompt.SETTLEMENT)
    if settlement_rows.any():
        settlement_mask = mask_t[settlement_rows, : settlement_logits.shape[1]]
        masked_logits = settlement_logits[settlement_rows].masked_fill(
            settlement_mask <= 0,
            -1e9,
        )
        preds[settlement_rows] = masked_logits.argmax(dim=-1)

    road_rows = prompt_t == int(PlacementPrompt.ROAD)
    if road_rows.any():
        road_mask = mask_t[road_rows, : road_logits.shape[1]]
        masked_logits = road_logits[road_rows].masked_fill(road_mask <= 0, -1e9)
        preds[road_rows] = masked_logits.argmax(dim=-1)

    return preds


def _accuracy_or_zero(preds, targets):
    if len(preds) == 0:
        return 0.0
    return (preds == targets).float().mean().item()


def _evaluate(model, obs_t, prompt_t, mask_t, act_t, weight_t, winner_flag_t):
    if len(obs_t) == 0:
        return {
            "loss": float("inf"),
            "acc": 0.0,
            "winner_acc": 0.0,
            "loser_acc": 0.0,
        }

    model.eval()
    with torch.no_grad():
        log_probs, _ = _log_probs_for_batch(model, obs_t, prompt_t, mask_t, act_t)
        loss = (-log_probs * weight_t).mean().item()
        preds = _predict_local_actions(model, obs_t, prompt_t, mask_t)
        acc = _accuracy_or_zero(preds, act_t)

        winner_acc = _accuracy_or_zero(preds[winner_flag_t], act_t[winner_flag_t])
        loser_acc = _accuracy_or_zero(preds[~winner_flag_t], act_t[~winner_flag_t])

    return {
        "loss": loss,
        "acc": acc,
        "winner_acc": winner_acc,
        "loser_acc": loser_acc,
    }


def train(
    *,
    chunk_paths,
    out_path,
    resume_path=None,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_size=64,
    val_frac=0.1,
    selection_mode="winner_only",
    win_weight=1.0,
    loss_weight=0.1,
    split_seed=0,
    metrics_path=None,
):
    chunk_paths = _resolve_chunk_paths(chunk_paths)
    (
        xs,
        prompts,
        masks,
        targets,
        weights,
        winner_flags,
        games_loaded,
    ) = _load_examples(chunk_paths, selection_mode, win_weight, loss_weight)

    device = get_device()
    model = PlacementModel(
        obs_size=COMPACT_PLACEMENT_FEATURE_SIZE,
        hidden_size=hidden_size,
    ).to(device)
    if resume_path:
        model.load_state_dict(
            torch.load(resume_path, map_location=device, weights_only=True)
        )

    rng = np.random.default_rng(split_seed)
    n = len(xs)
    perm = rng.permutation(n)
    split = int(n * (1 - val_frac))
    train_idx, val_idx = perm[:split], perm[split:]

    obs_t = torch.as_tensor(xs, dtype=torch.float32, device=device)
    prompt_t = torch.as_tensor(prompts, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(targets, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    winner_flag_t = torch.as_tensor(winner_flags, dtype=torch.bool, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {len(chunk_paths)} chunk files, {games_loaded} games, {n} examples\n"
        f"  selection_mode={selection_mode}\n"
        f"  train_examples={len(train_idx)}  val_examples={len(val_idx)}\n"
        f"  model_params={params:,}  device={device}",
        flush=True,
    )

    _append_metrics(
        metrics_path,
        {
            "event": "run_started",
            "chunk_paths": chunk_paths,
            "games_loaded": games_loaded,
            "num_examples": n,
            "train_examples": int(len(train_idx)),
            "val_examples": int(len(val_idx)),
            "selection_mode": selection_mode,
            "win_weight": win_weight,
            "loss_weight": loss_weight,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_size": hidden_size,
            "val_frac": val_frac,
            "split_seed": split_seed,
        },
    )

    best_val_loss = float("inf")
    best_state = None
    final_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        batch_perm = torch.as_tensor(rng.permutation(len(train_idx)), dtype=torch.long)
        running_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), batch_size):
            batch_rows = train_idx[batch_perm[start : start + batch_size]]
            b_obs = obs_t[batch_rows]
            b_prompt = prompt_t[batch_rows]
            b_mask = mask_t[batch_rows]
            b_act = act_t[batch_rows]
            b_weight = weight_t[batch_rows]

            log_probs, _ = _log_probs_for_batch(model, b_obs, b_prompt, b_mask, b_act)
            loss = (-log_probs * b_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        vi = torch.as_tensor(val_idx, dtype=torch.long, device=device)
        val_metrics = _evaluate(
            model,
            obs_t[vi],
            prompt_t[vi],
            mask_t[vi],
            act_t[vi],
            weight_t[vi],
            winner_flag_t[vi],
        )
        final_metrics = val_metrics

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        train_avg = running_loss / max(n_batches, 1)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_avg:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.2%}  "
            f"val_winner_acc={val_metrics['winner_acc']:.2%}  "
            f"val_loser_acc={val_metrics['loser_acc']:.2%}",
            flush=True,
        )
        _append_metrics(
            metrics_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_avg,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_winner_acc": val_metrics["winner_acc"],
                "val_loser_acc": val_metrics["loser_acc"],
                "best_val_loss": min(best_val_loss, val_metrics["loss"]),
            },
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = _save_model(model, out_path)
    print(f"\nBest model saved to {out_path}  (checkpoint: {ckpt_path})", flush=True)
    _append_metrics(
        metrics_path,
        {
            "event": "run_finished",
            "best_val_loss": best_val_loss,
            "final_val_loss": None if final_metrics is None else final_metrics["loss"],
            "out_path": out_path,
            "checkpoint_path": ckpt_path,
        },
    )
    return {
        "chunk_paths": list(chunk_paths),
        "games_loaded": games_loaded,
        "num_examples": n,
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "selection_mode": selection_mode,
        "best_val_loss": float(best_val_loss),
        "final_val_loss": None if final_metrics is None else float(final_metrics["loss"]),
        "final_val_acc": None if final_metrics is None else float(final_metrics["acc"]),
        "final_val_winner_acc": None if final_metrics is None else float(final_metrics["winner_acc"]),
        "final_val_loser_acc": None if final_metrics is None else float(final_metrics["loser_acc"]),
        "out_path": out_path,
        "checkpoint_path": ckpt_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train compact placement model from chunked supervised data"
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more compact chunk files or directories containing chunk files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="capstone_agent/models/placement_model.pt",
        help="Where to save trained placement weights",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional placement checkpoint to resume from",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="winner_only",
        choices=["winner_only", "outcome_weighted", "all_examples"],
        help="How to include winner/loser actions during training",
    )
    parser.add_argument("--win-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--metrics-log",
        type=str,
        default=None,
        help="Optional JSONL metrics log path",
    )
    args = parser.parse_args()

    train(
        chunk_paths=args.data,
        out_path=args.out,
        resume_path=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        val_frac=args.val_frac,
        selection_mode=args.selection_mode,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        split_seed=args.split_seed,
        metrics_path=args.metrics_log,
    )


if __name__ == "__main__":
    main()
