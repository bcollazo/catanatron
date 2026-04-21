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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CONSTANTS import PLACEMENT_AGENT_HIDDEN_SIZE
from PlacementGNNModel import PlacementGNNModel
from PlacementModel import PlacementModel
from device import get_device
from placement_action_space import PlacementPrompt
from placement_features import (
    COMPACT_NODE_FEATURE_SIZE,
    COMPACT_PLACEMENT_FEATURE_SIZE,
    NUM_NODES,
    STEP_INDICATOR_SIZE,
)
from placement_supervised_dataset import (
    load_chunk_records,
    iter_reconstructed_examples,
)


def _save_model(model, out_path):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    io_model = _model_for_state_io(model)

    tmp = out_path + ".tmp"
    torch.save(io_model.state_dict(), tmp)
    os.replace(tmp, out_path)

    root, ext = os.path.splitext(out_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    ckpt_path = f"{root}_{stamp}{ext}"
    tmp_ckpt = ckpt_path + ".tmp"
    torch.save(io_model.state_dict(), tmp_ckpt)
    os.replace(tmp_ckpt, ckpt_path)
    return ckpt_path


def _model_for_state_io(model):
    return getattr(model, "_orig_mod", model)


def _clone_model_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in _model_for_state_io(model).state_dict().items()
    }


def _training_state_path(weights_path: str) -> str:
    return weights_path + ".training_state.pt"


def _save_training_state(weights_path: str, training_state: dict) -> str:
    path = _training_state_path(weights_path)
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    tmp = path + ".tmp"
    torch.save(training_state, tmp)
    os.replace(tmp, path)
    return path


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
    xs, prompts, masks, targets, weights, winner_flags, game_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
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
                game_ids.append(str(record["game_id"]))

    if not xs:
        raise ValueError("No supervised examples were reconstructed from the input chunks")

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(prompts, dtype=np.int64),
        np.asarray(masks, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np.asarray(winner_flags, dtype=np.bool_),
        np.asarray(game_ids),
        games_loaded,
    )


def _flat_obs_to_graph(obs_t):
    """Reshape flat compact observations into structured GNN inputs."""

    batch_size = obs_t.shape[0]
    node_block = obs_t[:, : NUM_NODES * COMPACT_NODE_FEATURE_SIZE]
    node_features = node_block.reshape(
        batch_size,
        NUM_NODES,
        COMPACT_NODE_FEATURE_SIZE,
    )
    step_indicators = obs_t[:, -STEP_INDICATOR_SIZE:]
    return node_features, step_indicators


def _forward_model(model, obs_t, model_type):
    if model_type == "gnn":
        node_features, step_indicators = _flat_obs_to_graph(obs_t)
        return model(node_features, step_indicators)
    return model(obs_t)


def _apply_action_mask(logits, action_mask):
    fill_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(action_mask <= 0, fill_value)


def _log_probs_for_batch(model, obs_t, prompt_t, mask_t, act_t, model_type):
    settlement_logits, road_logits, _ = _forward_model(model, obs_t, model_type)
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
        masked_logits = _apply_action_mask(logits[rows], prompt_mask)
        dist = Categorical(logits=masked_logits)
        log_probs[rows] = dist.log_prob(act_t[rows])
        entropies[rows] = dist.entropy()

    return log_probs, entropies


def _predict_local_actions(model, obs_t, prompt_t, mask_t, model_type):
    settlement_logits, road_logits, _ = _forward_model(model, obs_t, model_type)
    preds = torch.empty(len(obs_t), dtype=torch.long, device=obs_t.device)

    settlement_rows = prompt_t == int(PlacementPrompt.SETTLEMENT)
    if settlement_rows.any():
        settlement_mask = mask_t[settlement_rows, : settlement_logits.shape[1]]
        masked_logits = _apply_action_mask(
            settlement_logits[settlement_rows],
            settlement_mask,
        )
        preds[settlement_rows] = masked_logits.argmax(dim=-1)

    road_rows = prompt_t == int(PlacementPrompt.ROAD)
    if road_rows.any():
        road_mask = mask_t[road_rows, : road_logits.shape[1]]
        masked_logits = _apply_action_mask(road_logits[road_rows], road_mask)
        preds[road_rows] = masked_logits.argmax(dim=-1)

    return preds


def _accuracy_or_zero(preds, targets):
    if len(preds) == 0:
        return 0.0
    return (preds == targets).float().mean().item()


def _evaluate(
    model,
    obs_t,
    prompt_t,
    mask_t,
    act_t,
    weight_t,
    winner_flag_t,
    model_type,
    use_amp,
    amp_device_type,
):
    if len(obs_t) == 0:
        return {
            "loss": float("inf"),
            "acc": 0.0,
            "winner_acc": 0.0,
            "loser_acc": 0.0,
            "settlement_acc": 0.0,
            "road_acc": 0.0,
        }

    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=amp_device_type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            log_probs, _ = _log_probs_for_batch(
                model,
                obs_t,
                prompt_t,
                mask_t,
                act_t,
                model_type,
            )
            loss = (-log_probs * weight_t).mean().item()
            preds = _predict_local_actions(model, obs_t, prompt_t, mask_t, model_type)
        acc = _accuracy_or_zero(preds, act_t)

        winner_acc = _accuracy_or_zero(preds[winner_flag_t], act_t[winner_flag_t])
        loser_acc = _accuracy_or_zero(preds[~winner_flag_t], act_t[~winner_flag_t])
        settlement_rows_mask = prompt_t == int(PlacementPrompt.SETTLEMENT)
        road_rows_mask = prompt_t == int(PlacementPrompt.ROAD)
        settlement_acc = _accuracy_or_zero(
            preds[settlement_rows_mask],
            act_t[settlement_rows_mask],
        )
        road_acc = _accuracy_or_zero(
            preds[road_rows_mask],
            act_t[road_rows_mask],
        )

    return {
        "loss": loss,
        "acc": acc,
        "winner_acc": winner_acc,
        "loser_acc": loser_acc,
        "settlement_acc": settlement_acc,
        "road_acc": road_acc,
    }


def train(
    *,
    chunk_paths,
    out_path,
    resume_path=None,
    epochs=50,
    batch_size=512,
    lr=3e-4,
    weight_decay=1e-4,
    hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
    val_frac=0.1,
    selection_mode="winner_only",
    win_weight=1.0,
    loss_weight=0.1,
    split_seed=0,
    metrics_path=None,
    patience=10,
    model_type="mlp",
    no_compile=False,
    no_amp=False,
):
    chunk_paths = _resolve_chunk_paths(chunk_paths)
    model_type = str(model_type).lower()
    if model_type not in {"mlp", "gnn"}:
        raise ValueError("model_type must be either 'mlp' or 'gnn'")

    (
        xs,
        prompts,
        masks,
        targets,
        weights,
        winner_flags,
        game_ids,
        games_loaded,
    ) = _load_examples(chunk_paths, selection_mode, win_weight, loss_weight)

    device = get_device()
    if model_type == "gnn":
        model = PlacementGNNModel(hidden_dim=hidden_size).to(device)
    else:
        model = PlacementModel(
            obs_size=COMPACT_PLACEMENT_FEATURE_SIZE,
            hidden_size=hidden_size,
        ).to(device)
    if resume_path:
        _model_for_state_io(model).load_state_dict(
            torch.load(resume_path, map_location=device, weights_only=True)
        )

    compile_warning = None
    if not no_compile and hasattr(torch, "compile"):
        if device.type == "mps" and model_type == "gnn":
            compile_warning = (
                "torch.compile() skipped for GNN on MPS due to operator support risk"
            )
        else:
            try:
                model = torch.compile(model)
                print("Model compiled with torch.compile()", flush=True)
            except Exception as exc:
                compile_warning = (
                    "torch.compile() not available, continuing without: "
                    f"{exc}"
                )
    if compile_warning is not None:
        print(compile_warning, flush=True)

    rng = np.random.default_rng(split_seed)
    n = len(xs)
    unique_game_ids = np.unique(game_ids)
    game_perm = rng.permutation(len(unique_game_ids))
    split = int(len(unique_game_ids) * (1 - val_frac))
    train_game_ids = set(unique_game_ids[game_perm[:split]])
    val_game_ids = set(unique_game_ids[game_perm[split:]])
    overlap = train_game_ids & val_game_ids
    if overlap:
        raise ValueError(f"Per-game split overlap detected: {sorted(overlap)!r}")

    train_idx = np.asarray(
        [idx for idx, game_id in enumerate(game_ids) if game_id in train_game_ids],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [idx for idx, game_id in enumerate(game_ids) if game_id in val_game_ids],
        dtype=np.int64,
    )

    obs_t = torch.as_tensor(xs, dtype=torch.float32, device=device)
    prompt_t = torch.as_tensor(prompts, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(targets, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    winner_flag_t = torch.as_tensor(winner_flags, dtype=torch.bool, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_epoch = 1
    best_val_loss = float("inf")
    best_state = None
    best_training_state = None
    final_metrics = None
    epochs_without_improvement = 0
    last_completed_epoch = start_epoch - 1

    if resume_path:
        resume_training_state_path = _training_state_path(resume_path)
        if os.path.exists(resume_training_state_path):
            training_state = torch.load(
                resume_training_state_path,
                map_location=device,
                weights_only=True,
            )
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            best_val_loss = float(training_state["best_val_loss"])
            epochs_without_improvement = int(
                training_state["epochs_without_improvement"]
            )
            start_epoch = int(training_state["epoch"]) + 1
            best_state = _clone_model_state_dict(model)
            best_training_state = training_state
            print(
                "Restored training state from "
                f"{resume_training_state_path} "
                f"(start_epoch={start_epoch}, best_val_loss={best_val_loss:.4f})",
                flush=True,
            )
        else:
            print(
                "No training state found at "
                f"{resume_training_state_path}; "
                "resuming with fresh optimizer/scheduler state from epoch 1",
                flush=True,
            )

    last_completed_epoch = start_epoch - 1

    params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {len(chunk_paths)} chunk files, {games_loaded} games, {n} examples\n"
        f"  selection_mode={selection_mode}\n"
        f"  model_type={model_type}\n"
        f"  compile_enabled={not no_compile}\n"
        f"  train_games={len(train_game_ids)}  val_games={len(val_game_ids)}\n"
        f"  train_examples={len(train_idx)}  val_examples={len(val_idx)}\n"
        f"  model_params={params:,}  device={device}",
        flush=True,
    )

    use_amp = device.type in ("cuda", "mps") and not no_amp
    amp_device_type = device.type
    if use_amp and device.type == "mps" and model_type == "gnn":
        print(
            "Automatic mixed precision disabled for GNN on MPS "
            "due to scatter_add_ float16 support risk",
            flush=True,
        )
        use_amp = False

    _append_metrics(
        metrics_path,
        {
            "event": "run_started",
            "chunk_paths": chunk_paths,
            "games_loaded": games_loaded,
            "num_examples": n,
            "train_games": int(len(train_game_ids)),
            "val_games": int(len(val_game_ids)),
            "train_examples": int(len(train_idx)),
            "val_examples": int(len(val_idx)),
            "selection_mode": selection_mode,
            "win_weight": win_weight,
            "loss_weight": loss_weight,
            "model_type": model_type,
            "epochs": epochs,
            "start_epoch": start_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_size": hidden_size,
            "val_frac": val_frac,
            "split_seed": split_seed,
            "patience": patience,
            "no_compile": no_compile,
            "no_amp": no_amp,
        },
    )

    for epoch in range(start_epoch, epochs + 1):
        last_completed_epoch = epoch
        model.train()
        batch_perm = torch.as_tensor(
            rng.permutation(len(train_idx_t)),
            dtype=torch.long,
            device=device,
        )
        running_loss = 0.0
        running_grad_norm_sum = 0.0
        max_grad_norm = 0.0
        n_batches = 0

        for start in range(0, len(train_idx_t), batch_size):
            batch_rows = train_idx_t[batch_perm[start : start + batch_size]]
            b_obs = obs_t[batch_rows]
            b_prompt = prompt_t[batch_rows]
            b_mask = mask_t[batch_rows]
            b_act = act_t[batch_rows]
            b_weight = weight_t[batch_rows]

            with torch.autocast(
                device_type=amp_device_type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                log_probs, _ = _log_probs_for_batch(
                    model,
                    b_obs,
                    b_prompt,
                    b_mask,
                    b_act,
                    model_type,
                )
                loss = (-log_probs * b_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_loss += loss.item()
            gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            running_grad_norm_sum += gn
            max_grad_norm = max(max_grad_norm, gn)
            n_batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        with torch.no_grad():
            max_train_eval = 2000
            if len(train_idx_t) == 0:
                train_acc = 0.0
            else:
                if len(train_idx_t) > max_train_eval:
                    subsample = rng.choice(
                        len(train_idx_t),
                        max_train_eval,
                        replace=False,
                    )
                    subsample_t = torch.as_tensor(
                        subsample,
                        dtype=torch.long,
                        device=device,
                    )
                    ti_eval = train_idx_t[subsample_t]
                else:
                    ti_eval = train_idx_t
                with torch.autocast(
                    device_type=amp_device_type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    train_preds = _predict_local_actions(
                        model,
                        obs_t[ti_eval],
                        prompt_t[ti_eval],
                        mask_t[ti_eval],
                        model_type,
                    )
                train_acc = (train_preds == act_t[ti_eval]).float().mean().item()
        model.train()

        val_metrics = _evaluate(
            model,
            obs_t[val_idx_t],
            prompt_t[val_idx_t],
            mask_t[val_idx_t],
            act_t[val_idx_t],
            weight_t[val_idx_t],
            winner_flag_t[val_idx_t],
            model_type,
            use_amp,
            amp_device_type,
        )
        final_metrics = val_metrics

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = _clone_model_state_dict(model)
            epochs_without_improvement = 0
            best_training_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }
            _save_training_state(
                out_path,
                best_training_state,
            )
        else:
            epochs_without_improvement += 1

        train_avg = running_loss / max(n_batches, 1)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_avg:.4f}  train_acc={train_acc:.2%}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.2%}  "
            f"settle_acc={val_metrics['settlement_acc']:.2%}  "
            f"road_acc={val_metrics['road_acc']:.2%}  "
            f"winner_acc={val_metrics['winner_acc']:.2%}",
            flush=True,
        )
        _append_metrics(
            metrics_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_avg,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_winner_acc": val_metrics["winner_acc"],
                "val_loser_acc": val_metrics["loser_acc"],
                "val_settlement_acc": val_metrics["settlement_acc"],
                "val_road_acc": val_metrics["road_acc"],
                "best_val_loss": min(best_val_loss, val_metrics["loss"]),
                "lr": current_lr,
                "mean_grad_norm": running_grad_norm_sum / max(n_batches, 1),
                "max_grad_norm": max_grad_norm,
            },
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)",
                flush=True,
            )
            break

    if best_state is not None:
        _model_for_state_io(model).load_state_dict(best_state)

    ckpt_path = _save_model(model, out_path)
    _save_training_state(
        out_path,
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": last_completed_epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        },
    )
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
        "train_games": int(len(train_game_ids)),
        "val_games": int(len(val_game_ids)),
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "selection_mode": selection_mode,
        "model_type": model_type,
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=PLACEMENT_AGENT_HIDDEN_SIZE)
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["mlp", "gnn"],
        help="Placement model architecture to train",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile() optimization",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
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
        patience=args.patience,
        model_type=args.model_type,
        no_compile=args.no_compile,
        no_amp=args.no_amp,
    )


if __name__ == "__main__":
    main()
