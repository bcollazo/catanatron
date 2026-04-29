# Codex Plan: Training Metrics Enhancement + Visualization Script

## Context

The placement model training pipeline (`capstone_agent/Placement/train_compact_placement_supervised.py`) already logs per-epoch metrics to a JSONL file via the `--metrics-log` flag. Each line is a JSON object with an `"event"` field: `"run_started"`, `"epoch"`, or `"run_finished"`.

Current `"epoch"` rows contain: `train_loss`, `val_loss`, `val_acc`, `val_winner_acc`, `val_loser_acc`, `best_val_loss`.

There are two deliverables:
1. **Enrich the logged metrics** in the training script (add 5 new fields per epoch row)
2. **Create a new visualization script** that reads any JSONL metrics file and produces publication-quality plots

---

## Change 1: Enrich epoch metrics in the training script

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

### 1a. Log train accuracy

In the training loop (around lines 403-436), after the inner batch loop finishes, run a quick no-grad accuracy pass over the training set. This is expensive for huge datasets, so do it on a **random subsample** of at most 2000 training examples.

Add this right after `scheduler.step()` (line 436), before the existing `_evaluate()` call on the val set:

```python
# --- train accuracy on a subsample ---
model.eval()
with torch.no_grad():
    max_train_eval = 2000
    if len(train_idx) > max_train_eval:
        subsample = rng.choice(len(train_idx), max_train_eval, replace=False)
        ti_eval = torch.as_tensor(train_idx[subsample], dtype=torch.long, device=device)
    else:
        ti_eval = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    train_preds = _predict_local_actions(
        model, obs_t[ti_eval], prompt_t[ti_eval], mask_t[ti_eval], model_type
    )
    train_acc = (train_preds == act_t[ti_eval]).float().mean().item()
model.train()
```

Include `"train_acc": train_acc` in the epoch metrics dict (the `_append_metrics` call around line 481).

### 1b. Log settlement vs road accuracy (val set)

Modify the `_evaluate` function (starts around line 211) to also return `settlement_acc` and `road_acc`. Inside `_evaluate`, after computing `preds` via `_predict_local_actions`, split by prompt type:

```python
settlement_rows_mask = prompt_t == int(PlacementPrompt.SETTLEMENT)
road_rows_mask = prompt_t == int(PlacementPrompt.ROAD)
settlement_acc = _accuracy_or_zero(preds[settlement_rows_mask], act_t[settlement_rows_mask])
road_acc = _accuracy_or_zero(preds[road_rows_mask], act_t[road_rows_mask])
```

Add `"settlement_acc"` and `"road_acc"` to the returned dict. Then include `val_metrics["settlement_acc"]` and `val_metrics["road_acc"]` in both the epoch metrics dict and the print statement.

**Important:** `PlacementPrompt` is already imported in this file (from `placement_action_space`). The `prompt_t` tensor is already passed to `_evaluate` as the second positional argument.

Wait — looking at the current signature:

```python
def _evaluate(model, obs_t, prompt_t, mask_t, act_t, weight_t, winner_flag_t, model_type):
```

`prompt_t` is the second argument after `model`. So you have access to it. Use it to split preds by prompt type.

### 1c. Log learning rate

After `scheduler.step()`, log the current LR:

```python
current_lr = scheduler.get_last_lr()[0]
```

Include `"lr": current_lr` in the epoch metrics dict.

### 1d. Log gradient norm

`torch.nn.utils.clip_grad_norm_` (line 430) returns the total gradient norm **before** clipping. Capture it:

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
```

Accumulate the max and mean grad norm across batches in the epoch:

```python
# Before the batch loop:
running_grad_norm_sum = 0.0
max_grad_norm = 0.0

# Inside the batch loop, after clip_grad_norm_:
gn = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
running_grad_norm_sum += gn
max_grad_norm = max(max_grad_norm, gn)
```

Include `"mean_grad_norm": running_grad_norm_sum / max(n_batches, 1)` and `"max_grad_norm": max_grad_norm` in the epoch metrics dict.

### 1e. Update the print statement

Update the per-epoch print (around line 472) to also show settlement/road accuracy:

```python
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
```

### Summary of new fields in each `"epoch"` row

After this change, each epoch row will contain:

| Field | Source | New? |
|---|---|---|
| `epoch` | loop counter | existing |
| `train_loss` | running average | existing |
| `train_acc` | subsample eval | **NEW** |
| `val_loss` | `_evaluate` | existing |
| `val_acc` | `_evaluate` | existing |
| `val_winner_acc` | `_evaluate` | existing |
| `val_loser_acc` | `_evaluate` | existing |
| `val_settlement_acc` | `_evaluate` | **NEW** |
| `val_road_acc` | `_evaluate` | **NEW** |
| `best_val_loss` | running min | existing |
| `lr` | scheduler | **NEW** |
| `mean_grad_norm` | clip_grad_norm_ | **NEW** |
| `max_grad_norm` | clip_grad_norm_ | **NEW** |

---

## Change 2: Create the visualization script

**New file:** `capstone_agent/Placement/plot_training_metrics.py`

This script reads one or more JSONL metrics files and produces matplotlib plots.

### CLI interface

```
python capstone_agent/Placement/plot_training_metrics.py \
    path/to/training_metrics.jsonl \
    [path/to/another_run.jsonl ...] \
    --out-dir plots/ \
    --format png
```

Arguments:
- Positional: one or more JSONL file paths
- `--out-dir`: directory to save plots (default: same directory as the first JSONL file)
- `--format`: `png` or `pdf` (default: `png`)
- `--dpi`: resolution (default: 150)
- `--no-show`: suppress `plt.show()`, just save files

### Parsing logic

```python
def load_metrics(jsonl_path):
    """Return (run_meta, epoch_rows) where run_meta is the run_started dict
    and epoch_rows is a list of epoch dicts, sorted by epoch number."""
    run_meta = {}
    epochs = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if row["event"] == "run_started":
                run_meta = row
            elif row["event"] == "epoch":
                epochs.append(row)
    # Sort by epoch in case of appended resumed runs
    epochs.sort(key=lambda r: r["epoch"])
    return run_meta, epochs
```

If a file contains multiple `run_started` events (from resumed training), keep the first one as `run_meta` and note the epoch numbers where each new `run_started` occurs (for vertical resume markers).

### Run label

Auto-generate a short label from `run_meta`:

```python
def make_label(run_meta, path):
    model_type = run_meta.get("model_type", "mlp").upper()
    games = run_meta.get("games_loaded", "?")
    hidden = run_meta.get("hidden_size", "?")
    lr = run_meta.get("lr", "?")
    return f"{model_type} g={games} h={hidden} lr={lr}"
```

When multiple files are provided, each gets its own color and label in legends.

### Plot functions

Implement each plot as a separate function. All functions should:
- Accept a list of `(label, run_meta, epoch_rows)` tuples for multi-run overlay
- Use a consistent color palette (e.g., tab10)
- Include grid lines, axis labels, titles, and legends
- Return the figure object

#### Plot 1: `plot_loss_curves(runs, out_dir, fmt)`
- X-axis: epoch
- Y-axis: loss
- For each run: solid line for `train_loss`, dashed line for `val_loss`, same color
- Dotted horizontal line for `best_val_loss` (final value)
- If `best_val_loss` improves at some epoch, mark that epoch with a star on the val_loss line
- Title: "Training & Validation Loss"
- Save as `loss_curves.{fmt}`

#### Plot 2: `plot_accuracy_curves(runs, out_dir, fmt)`
- X-axis: epoch
- Y-axis: accuracy (0 to 1, formatted as %)
- For each run: lines for `val_acc`, `val_winner_acc`, `val_loser_acc`
- If `val_settlement_acc` and `val_road_acc` exist in the data, plot those too (with different line styles)
- If `train_acc` exists, plot it as a dotted line
- Horizontal dashed line at 0.20 labeled "random baseline"
- Title: "Accuracy Over Training"
- Save as `accuracy_curves.{fmt}`

#### Plot 3: `plot_overfitting_gap(runs, out_dir, fmt)`
- X-axis: epoch
- Y-axis: gap value
- For each run: plot `val_loss - train_loss`
- If `train_acc` exists: secondary y-axis showing `train_acc - val_acc`
- Horizontal line at 0
- Title: "Generalization Gap (val_loss − train_loss)"
- Save as `overfitting_gap.{fmt}`

#### Plot 4: `plot_lr_and_grad_norm(runs, out_dir, fmt)`
- Only generate this plot if `lr` exists in the epoch data
- X-axis: epoch
- Left y-axis: learning rate (log scale if it varies by more than 10x)
- Right y-axis: `mean_grad_norm` as a line, `max_grad_norm` as a faint filled region
- Title: "Learning Rate & Gradient Norm"
- Save as `lr_grad_norm.{fmt}`

#### Plot 5: `plot_settlement_vs_road(runs, out_dir, fmt)`
- Only generate this plot if `val_settlement_acc` exists in the epoch data
- X-axis: epoch
- Y-axis: accuracy
- For each run: solid line for `val_settlement_acc`, dashed for `val_road_acc`
- Horizontal dashed line at random baseline
- Title: "Settlement vs Road Accuracy"
- Save as `settlement_vs_road.{fmt}`

#### Plot 6: `plot_dashboard(runs, out_dir, fmt)`
- A 2x2 subplot figure combining:
  - Top-left: loss curves (Plot 1 logic)
  - Top-right: accuracy curves (Plot 2 logic, simplified — just val_acc and train_acc)
  - Bottom-left: overfitting gap (Plot 3 logic)
  - Bottom-right: settlement vs road (Plot 5 logic) if available, else LR schedule (Plot 4 logic)
- `fig.suptitle()` with the run label and key hyperparams
- `fig.tight_layout()`
- Save as `training_dashboard.{fmt}`
- This is the single most useful output — the "one picture to rule them all"

### Main function

```python
def main():
    parser = argparse.ArgumentParser(description="Plot placement training metrics")
    parser.add_argument("metrics", nargs="+", help="JSONL metrics file(s)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.metrics[0]) or "."
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for path in args.metrics:
        meta, epochs = load_metrics(path)
        label = make_label(meta, path)
        runs.append((label, meta, epochs))

    plot_loss_curves(runs, out_dir, args.format, args.dpi)
    plot_accuracy_curves(runs, out_dir, args.format, args.dpi)
    plot_overfitting_gap(runs, out_dir, args.format, args.dpi)
    plot_lr_and_grad_norm(runs, out_dir, args.format, args.dpi)
    plot_settlement_vs_road(runs, out_dir, args.format, args.dpi)
    plot_dashboard(runs, out_dir, args.format, args.dpi)

    if not args.no_show:
        plt.show()
```

### Style requirements

- Use `matplotlib` only (no seaborn dependency)
- Set `plt.style.use("seaborn-v0_8-whitegrid")` at module level (fall back to `"ggplot"` if not available)
- Figure size: individual plots 10x6, dashboard 14x10
- All plots should be readable in grayscale (use line styles, not just colors, to differentiate series)
- Include `matplotlib.ticker.PercentFormatter(1.0)` on accuracy y-axes
- Font size: titles 14pt, axis labels 12pt, tick labels 10pt

### Graceful handling

- If a field doesn't exist in the epoch data (e.g., `train_acc` in older JSONL files), skip the series silently and note it in the legend as "(not available)"
- If only one run is provided, don't add a run-label prefix to series names
- If `--out-dir` doesn't exist, create it

---

## Change 3: Add `--results-log` to benchmark script

**File:** `capstone_agent/Placement/benchmark_placement.py`

Add an optional `--results-log` argument that writes a JSONL file with one row per completed benchmark group:

```python
parser.add_argument(
    "--results-log", type=str, default=None,
    help="Optional JSONL file to append benchmark results",
)
```

After each group finishes (baseline and test), append a JSON row:

```python
{
    "event": "benchmark_group",
    "label": label,
    "strategy": args.strategy,
    "placement_model": args.placement_model,
    "games": n,
    "wins": wins,
    "losses": losses,
    "draws": draws,
    "win_rate": wins / n,
    "ci_lo": lo,
    "ci_hi": hi,
    "elapsed_seconds": elapsed,
    "timestamp": datetime.now(timezone.utc).isoformat()
}
```

This makes benchmark results programmatically accessible for future comparison plots.

---

## Verification

1. `python -m py_compile capstone_agent/Placement/train_compact_placement_supervised.py` — should pass
2. `python -m py_compile capstone_agent/Placement/plot_training_metrics.py` — should pass
3. `python -m py_compile capstone_agent/Placement/benchmark_placement.py` — should pass
4. Run the plot script against the existing old metrics file to confirm it handles missing fields gracefully:
   ```
   python capstone_agent/Placement/plot_training_metrics.py \
       capstone_agent/Placement/online_runs/compact_placement_20260402T1959Z/logs/training_metrics.jsonl \
       --out-dir /tmp/placement_plots_test \
       --no-show
   ```
   This should produce at least `loss_curves.png`, `accuracy_curves.png`, `overfitting_gap.png`, and `training_dashboard.png` without errors. The LR and settlement/road plots should either be skipped or show "(not available)" since the old file doesn't have those fields.
5. Verify the enriched training script still runs end-to-end by checking that the new fields would be present in the epoch dict (code inspection is fine — no need to run a full training job).
