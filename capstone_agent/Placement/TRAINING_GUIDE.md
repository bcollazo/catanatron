# Placement Agent: Data Collection, Training, and Benchmarking Guide

All commands assume you are in the project root: `/Users/peterb/Desktop/academia/catan_ai`

---

## Step 1: Collect Training Data

```bash
python capstone_agent/Placement/collect_compact_placement_data.py \
  --games 50000 \
  --fixed-board \
  --board-seed 42 \
  --ab-depth 2 \
  --out-dir capstone_agent/data/compact_placement_fixed \
  --seed 0
```

**What this does:** Plays 50,000 two-player Catan games. Both players place settlements and roads randomly, then AlphaBeta plays the rest. Records who won and what placements each player made. Saves the data as `.npz` chunk files.

**Parameters:**

| Flag | What it does | Default |
|------|-------------|---------|
| `--games` | Number of games to simulate | 5000 |
| `--fixed-board` | All games use the same board layout (hex/number/port arrangement). Use `--no-fixed-board` to randomize per game. | On |
| `--board-seed` | Which board layout to use. Same seed = same board. Only matters when `--fixed-board` is on. | 42 |
| `--ab-depth` | How many moves ahead AlphaBeta looks during the post-placement game. Higher = stronger but slower. | 2 |
| `--out-dir` | Where to write the `.npz` data files | `capstone_agent/data/compact_placement` |
| `--seed` | Base random seed. Game `i` uses seed `base + i`. | 0 |
| `--workers` | Number of CPU cores to use. | All but one |
| `--games-per-chunk` | How many games go into each `.npz` file | 5000 |

**How long it takes:** Roughly 1-3 games/second depending on your machine. 50,000 games takes several hours. You can Ctrl+C at any time -- completed chunks are saved.

**To add more data later**, run again with a different `--seed` so game seeds don't overlap:

```bash
# First batch: seeds 0 through 49999
python capstone_agent/Placement/collect_compact_placement_data.py \
  --games 50000 --seed 0 --fixed-board \
  --out-dir capstone_agent/data/compact_placement_fixed

# Second batch: seeds 50000 through 99999
python capstone_agent/Placement/collect_compact_placement_data.py \
  --games 50000 --seed 50000 --fixed-board \
  --out-dir capstone_agent/data/compact_placement_fixed
```

Both batches write to the same directory. Training picks up all `.npz` files in the directory automatically.

---

## Step 2: Train the Model

### MLP model (default)

```bash
python capstone_agent/Placement/train_compact_placement_supervised.py \
  --data capstone_agent/data/compact_placement_fixed \
  --out capstone_agent/models/placement_model.pt \
  --selection-mode outcome_weighted \
  --model-type mlp \
  --metrics-log capstone_agent/data/compact_placement_fixed/training_metrics.jsonl
```

### GNN model

```bash
python capstone_agent/Placement/train_compact_placement_supervised.py \
  --data capstone_agent/data/compact_placement_fixed \
  --out capstone_agent/models/placement_model_gnn.pt \
  --selection-mode outcome_weighted \
  --model-type gnn \
  --metrics-log capstone_agent/data/compact_placement_fixed/training_metrics_gnn.jsonl
```

**Parameters:**

| Flag | What it does | Default |
|------|-------------|---------|
| `--data` | Path to directory of `.npz` files (or list of individual files) | *required* |
| `--out` | Where to save the trained model weights | `capstone_agent/models/placement_model.pt` |
| `--resume` | Path to a previous checkpoint to continue training from | None (train from scratch) |
| `--model-type` | `mlp` (flat network) or `gnn` (graph neural network) | `mlp` |
| `--selection-mode` | Which examples to train on (see below) | `winner_only` |
| `--epochs` | Maximum training passes over the data | 50 |
| `--batch-size` | Examples per gradient update | 512 |
| `--lr` | Learning rate | 0.0003 |
| `--weight-decay` | L2 regularization strength | 0.0001 |
| `--hidden-size` | Width of hidden layers | 256 |
| `--patience` | Stop early if val loss doesn't improve for this many epochs | 10 |
| `--val-frac` | Fraction of games held out for validation | 0.1 |
| `--split-seed` | Random seed for the train/val split | 0 |
| `--win-weight` | Sample weight for winning-side examples | 1.0 |
| `--loss-weight` | Sample weight for losing-side examples (only used with `outcome_weighted`) | 0.1 |
| `--metrics-log` | Path to write per-epoch metrics as JSONL | None (no log) |

**Selection modes explained:**

- `winner_only` -- Only train on the 4 placement actions from the winning player in each game. Discards the loser's data entirely. 4 examples per game.
- `outcome_weighted` -- **Recommended.** Train on ALL 8 placement actions from both players, but weight winning actions at `--win-weight` (1.0) and losing actions at `--loss-weight` (0.1). Gives the model both positive and negative signal. 8 examples per game.
- `all_examples` -- Train on all 8 actions equally weighted. No win/loss distinction.

**What to watch during training:**

- `train_loss` should decrease
- `val_loss` should decrease (if it only goes up, the model is overfitting)
- `val_acc` should increase above 20% (20% is the random baseline)
- Early stopping will kick in if val loss doesn't improve for `--patience` epochs

---

## Step 3: Resume Training

If training was interrupted or you want to continue with more epochs:

```bash
python capstone_agent/Placement/train_compact_placement_supervised.py \
  --data capstone_agent/data/compact_placement_fixed \
  --out capstone_agent/models/placement_model.pt \
  --resume capstone_agent/models/placement_model.pt \
  --selection-mode outcome_weighted \
  --model-type mlp \
  --epochs 100
```

This loads the saved model weights and (if available) the optimizer/scheduler state from `placement_model.pt.training_state.pt`, then continues training. The `--epochs` value is the total epoch count, not additional epochs -- if the model was saved at epoch 30, it resumes from epoch 31 and runs through epoch 100.

---

## Step 4: Benchmark

Test how the trained placement model performs against AlphaBeta:

```bash
python capstone_agent/Placement/benchmark_placement.py \
  --games 200 \
  --strategy model \
  --placement-model capstone_agent/models/placement_model.pt \
  --verbose
```

This runs two groups of games:
1. **Baseline**: AlphaBeta placement + AlphaBeta rest (Blue) vs AlphaBeta (Red)
2. **Test**: Your trained model placement + AlphaBeta rest (Blue) vs AlphaBeta (Red)

It reports win rate and 95% confidence interval for each group, plus the delta.

**Parameters:**

| Flag | What it does | Default |
|------|-------------|---------|
| `--games` | Games per group | 100 |
| `--strategy` | `model` (your trained model) or `random` (baseline comparison) | `random` |
| `--placement-model` | Path to trained weights (required when strategy is `model`) | None |
| `--skip-baseline` | Skip the AlphaBeta-vs-AlphaBeta group | Off |
| `--verbose` | Print progress every 50 games | Off |

**What good results look like:**
- AlphaBeta baseline should win ~48% (it's roughly balanced playing itself)
- Random placement should win ~10-15%
- Your trained model should win >40% to be considered useful, >48% to match AlphaBeta

**Note:** The benchmark script currently uses `model_type="mlp"` by default. To benchmark a GNN model, you would need to add `model_type="gnn"` support to the benchmark script (it doesn't have a `--model-type` flag yet).

---

## Quick Reference: Full Pipeline

```bash
# 1. Collect 50k games (run overnight)
python capstone_agent/Placement/collect_compact_placement_data.py \
  --games 50000 --fixed-board --out-dir capstone_agent/data/compact_placement_fixed

# 2. Train
python capstone_agent/Placement/train_compact_placement_supervised.py \
  --data capstone_agent/data/compact_placement_fixed \
  --out capstone_agent/models/placement_model.pt \
  --selection-mode outcome_weighted \
  --metrics-log capstone_agent/data/compact_placement_fixed/metrics.jsonl

# 3. Benchmark
python capstone_agent/Placement/benchmark_placement.py \
  --games 200 --strategy model \
  --placement-model capstone_agent/models/placement_model.pt --verbose

# 4. (Optional) Collect more data and retrain
python capstone_agent/Placement/collect_compact_placement_data.py \
  --games 50000 --seed 50000 --fixed-board \
  --out-dir capstone_agent/data/compact_placement_fixed

python capstone_agent/Placement/train_compact_placement_supervised.py \
  --data capstone_agent/data/compact_placement_fixed \
  --out capstone_agent/models/placement_model.pt \
  --resume capstone_agent/models/placement_model.pt \
  --selection-mode outcome_weighted \
  --epochs 100
```
