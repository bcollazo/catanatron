# Placement Agent Guide

The placement stack now uses a compact opening-only representation and a
prompt-specific action space:

- node block: `54 x [self_settlement, opp_settlement, 5 resource pips, 6 port channels]`
- edge block: `72 x [self_road, opp_road]`
- policy heads: `54` settlement logits and `72` road logits

The opening agent stays actor-relative, then hands off to the main-game policy
through `CapstoneAgent`.

## Main Pieces

| Component | File | Role |
|---|---|---|
| `PlacementModel` | `PlacementModel.py` | Compact dual-head opening policy + value head |
| `PlacementAgent` | `PlacementAgent.py` | PPO/supervised wrapper for the compact model |
| `placement_features` | `placement_features.py` | Canonical compact feature assembly/projection |
| `placement_action_space` | `placement_action_space.py` | Prompt inference + local/global action mapping |
| `placement_supervised_dataset` | `placement_supervised_dataset.py` | Chunk schema, one-hot encoding, 8-ply reconstruction |
| `router_search_player` | `router_search_player.py` | Native engine bridge that routes through `CapstoneAgent` |
| `train_compact_placement_online` | `train_compact_placement_online.py` | Interleaved collector + snapshot trainer + periodic benchmark loop |

## Canonical Supervised Workflow

### Step 1: Collect chunked compact data

This is the canonical collector for the supervised opening pipeline. It runs
full games through a literal `CapstoneAgent` router:

- placement phase: `RandomPlacementAgent`
- main phase: `AlphaBeta`

```bash
python capstone_agent/collect_compact_placement_data.py \
    --games 5000 \
    --games-per-chunk 5000 \
    --out-dir capstone_agent/data/compact_placement
```

Each chunk stores one row per completed game:

- `static_node_features` `(G, 54, 11)`
- `opening_actions_onehot` `(G, 8, 126)`
- `winner_is_first_actor` `(G,)`
- metadata: `game_id`, `first_actor_color`, `winner_color`, `game_seed`
- `schema_version`

A sibling manifest `.jsonl` records chunk boundaries, counts, and run metadata.

### Step 2: Train from chunked data

```bash
python capstone_agent/train_compact_placement_supervised.py \
    --data capstone_agent/data/compact_placement \
    --epochs 30 \
    --selection-mode winner_only \
    --out capstone_agent/models/placement_model.pt \
    --metrics-log capstone_agent/models/placement_train_metrics.jsonl
```

Key points:

- reconstructs each `X` offline from `static_node_features + prior opening actions`
- rebuilds the legal local mask for all 8 opening plies
- defaults to `winner_only` supervision, but also supports
  `outcome_weighted` and `all_examples`
- writes per-epoch metrics to JSONL if `--metrics-log` is provided
- saves the best checkpoint by validation loss

### Step 3: Benchmark against AlphaBeta

```bash
python capstone_agent/benchmark_placement.py \
    --games 200 \
    --strategy model \
    --placement-model capstone_agent/models/placement_model.pt \
    --verbose
```

This keeps AlphaBeta for the rest of the game so the independent variable is
just the opening placement policy.

### Optional: Interleaved collect/train loop

If you want collection and supervised training to run continuously in the same
process, use the online coordinator:

```bash
python capstone_agent/train_compact_placement_online.py \
    --cycles 10 \
    --workers 8 \
    --collector-games-per-window 1000 \
    --collector-games-per-chunk 1000 \
    --min-new-chunks 1 \
    --replay-chunk-ratio 1.0 \
    --benchmark-games 100
```

This coordinator:

- launches repeated background collection windows into a run-scoped `data/` directory
- lets you set the collection CPU budget directly with `--workers N` (alias for `--collector-workers`)
- trains each cycle on all unseen chunks plus a sampled replay subset of older chunks
- resumes from the latest placement checkpoint each cycle instead of restarting from scratch
- writes persistent coordinator state to `run_dir/state.json` and event logs to `run_dir/logs/status.jsonl`
- benchmarks the latest checkpoint against full AlphaBeta after each configured cycle

## Legacy Flat Workflow

The old flat per-decision collector/trainer still exist for compatibility:

- `collect_placement_data.py`
- `train_placement.py`

They are now legacy utilities. `train_placement.py` has been updated so it is
not broken against the compact model, but new supervised work should prefer the
chunked compact pipeline above.

## Using The Placement Agent Elsewhere

`CapstoneAgent` still routes by phase exactly as before, so a trained compact
placement model remains a drop-in opening specialist for:

- `run_simulation.py`
- `training_loop.py`
- `benchmark_placement.py`

## Quick Reference

```bash
# Canonical compact SL pipeline
python capstone_agent/collect_compact_placement_data.py --games 5000 --games-per-chunk 5000 --out-dir capstone_agent/data/compact_placement
python capstone_agent/train_compact_placement_supervised.py --data capstone_agent/data/compact_placement --epochs 30 --selection-mode winner_only --out capstone_agent/models/placement_model.pt --metrics-log capstone_agent/models/placement_train_metrics.jsonl
python capstone_agent/benchmark_placement.py --games 200 --strategy model --placement-model capstone_agent/models/placement_model.pt --verbose

# Interleaved collect/train/benchmark loop
python capstone_agent/train_compact_placement_online.py --cycles 10 --workers 8 --collector-games-per-window 1000 --collector-games-per-chunk 1000 --min-new-chunks 1 --replay-chunk-ratio 1.0 --benchmark-games 100
```

## Hardware Acceleration

Training and inference automatically use the best available device:

- Apple Silicon: MPS
- NVIDIA GPU: CUDA
- Fallback: CPU
