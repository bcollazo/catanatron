# Placement Agent Guide

This branch adds a **separate initial-placement agent** that handles the first settlement and road decisions in Catan, then hands off to the main MainPlayAgent (or AlphaBeta) for the rest of the game.

## Architecture Overview

| Component | File | Role |
|---|---|---|
| `PlacementModel` | `PlacementModel.py` | Lightweight neural net (settlement + road heads only, ~880K params) |
| `PlacementAgent` | `PlacementAgent.py` | Wraps the model with PPO and supervised learning interfaces |
| `RandomPlacementAgent` | `PlacementAgent.py` | Baseline that picks uniformly from valid placement actions |
| `CapstoneAgent` | `CapstoneAgent.py` | Routes decisions to the placement agent during initial build, main agent otherwise |
| `device.py` | `device.py` | Shared device selection (MPS > CUDA > CPU) |

The placement agent can be swapped between strategies via `make_placement_agent("model")` or `make_placement_agent("random")`.

---

## Workflow: Collect Data → Train → Benchmark

### Step 1: Collect placement data

Run bot-vs-bot games and record every placement decision Blue makes, along with whether Blue won.

```bash
python capstone_agent/collect_placement_data.py --games 5000
```

Both Blue and Red default to AlphaBeta, so the recorded placements reflect real strategy. Progress prints automatically (~20 status updates).

**Output:** `capstone_agent/data/placement_data.npz` containing:
- `obs` (N, 1259) — board observations at each placement decision
- `masks` (N, 245) — valid action masks
- `actions` (N,) — the action index AlphaBeta chose
- `won` (N,) — 1.0 if Blue won that game, 0.0 otherwise

With 5000 games you get ~20,000 samples (4 placement decisions per game).

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--games` | 5000 | Number of games to simulate |
| `--blue` | `alphabeta` | Bot type for Blue (the player we learn from) |
| `--enemy` | `alphabeta` | Bot type for Red |
| `--out` | `capstone_agent/data/placement_data.npz` | Output path |

Available bot types: `alphabeta`, `alphabeta-prune`, `same-turn-ab`, `value`, `vp`, `weighted`, `random`.

**Time estimate:** AlphaBeta games take ~1s each, so 5000 games ≈ 80 minutes.

---

### Step 2: Train the placement model

Train the `PlacementModel` on the collected data using supervised learning (weighted cross-entropy loss).

```bash
python capstone_agent/train_placement.py \
    --data capstone_agent/placement_data.npz \
    --epochs 30
```

The training loop:
- Splits data into train/validation (90/10)
- Weights samples from winning games higher than losing games
- Uses cosine annealing on the learning rate
- Saves the best model (by validation loss)
- Prints per-epoch metrics: train loss, val loss, val accuracy, win-sample accuracy

**Output:** `capstone_agent/models/placement_model.pt` + a timestamped checkpoint (e.g. `placement_model_20260319T1506Z.pt`).

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--data` | `capstone_agent/data/placement_data.npz` | Path to the `.npz` dataset |
| `--out` | `capstone_agent/models/placement_model.pt` | Where to save trained weights |
| `--epochs` | 30 | Training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--lr` | 0.001 | Learning rate |
| `--win-weight` | 1.0 | Loss weight for samples from winning games |
| `--loss-weight` | 0.1 | Loss weight for samples from losing games |
| `--hidden-size` | 64 | Model hidden dimension |
| `--weight-decay` | 0.0001 | AdamW L2 regularization |
| `--val-frac` | 0.1 | Fraction of data reserved for validation |

---

### Step 3: Benchmark against AlphaBeta

Compare your placement strategy against the AlphaBeta baseline. Both groups use AlphaBeta for the rest of the game — the only independent variable is who chooses the initial placements.

**Benchmark a trained model:**

```bash
python capstone_agent/benchmark_placement.py \
    --games 200 \
    --strategy model \
    --placement-model capstone_agent/models/placement_model.pt \
    --verbose
```

**Benchmark random placement (sanity check):**

```bash
python capstone_agent/benchmark_placement.py \
    --games 200 \
    --strategy random \
    --verbose
```

**Output:** Win/loss/draw counts, win rates with 95% confidence intervals, and the delta between your strategy and the AlphaBeta baseline.

```
========================================================================
RESULTS
========================================================================
  AlphaBeta (baseline)         98W /  102L /    0D  win_rate=49.0%  95% CI=[42.1%, 55.9%]  (320s)
  model (placement_model.pt)  112W /   88L /    0D  win_rate=56.0%  95% CI=[49.1%, 62.9%]  (298s)

  Delta (test - baseline): +7.0%
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--games` | 100 | Games per group |
| `--strategy` | `random` | `random` or `model` |
| `--placement-model` | — | Path to `.pt` weights (required for `model`) |
| `--skip-baseline` | off | Skip the AlphaBeta-vs-AlphaBeta group |
| `--verbose` | off | Print progress every 50 games |

---

## Using the Placement Agent in Training

### With `run_simulation.py`

The main simulation script uses the `CapstoneAgent` by default. It routes initial placement to the placement agent and everything else to the MainPlayAgent.

```bash
# Train with a learned placement model
python capstone_agent/run_simulation.py --train --games 500 \
    --placement-model capstone_agent/models/placement_model.pt

# Train with random placement
python capstone_agent/run_simulation.py --train --games 500 \
    --placement-strategy random

# Evaluate only (no training)
python capstone_agent/run_simulation.py --games 50 \
    --load capstone_agent/models/capstone_model.pt \
    --placement-model capstone_agent/models/placement_model.pt
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--placement-strategy` | `model` | `model` or `random` |
| `--placement-model` | — | Path to placement weights |
| `--save-placement-model` | `capstone_agent/models/placement_model.pt` | Save path for placement weights after training |
| `--fresh-start` | off | Ignore existing saved weights |

Both the main model and placement model are saved with timestamped checkpoints to avoid overwriting previous versions.

### With `training_loop.py`

The simpler training loop also uses `CapstoneAgent` and saves both models:

```bash
python capstone_agent/training_loop.py
```

---

## Quick Reference

```bash
# Full pipeline: collect → train → benchmark
python capstone_agent/collect_placement_data.py --games 5000
python capstone_agent/train_placement.py --data capstone_agent/data/placement_data.npz --epochs 30
python capstone_agent/benchmark_placement.py --games 200 --strategy model \
    --placement-model capstone_agent/models/placement_model.pt --verbose
```

## Hardware Acceleration

All training and inference automatically uses the best available device:
- **Apple Silicon:** MPS (Metal Performance Shaders)
- **NVIDIA GPU:** CUDA
- **Fallback:** CPU

The device is printed at startup. No flags needed.
