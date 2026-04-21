# Codex Plan: Heuristic Benchmark + Value Function with 1-Ply Search

## Overview

Two sequential deliverables:

- **Part A**: Implement the heuristic placement agent (per the existing plan `CODEX_PLAN_HEURISTIC_PLACEMENT.md`) and benchmark it against AlphaBeta over 500 games. This measures how strong placement-level strategy is on its own.
- **Part B**: Train a value network `V(state) → P(win)`, wrap it in a 1-ply-search placement agent, and benchmark the same way. Goal: beat the heuristic.

Run Part A first. Look at the win rate. Then decide whether to proceed with Part B.

---

# PART A — Heuristic Agent and 500-Game Benchmark

## A1. Implement the heuristic (see existing plan)

**Follow `CODEX_PLAN_HEURISTIC_PLACEMENT.md` for Deliverables 1, 2, and 4 (the factory method).** Specifically:
- Create `capstone_agent/Placement/placement_heuristic.py` with the scoring functions.
- Add `HeuristicPlacementAgent` class to `capstone_agent/Placement/PlacementAgent.py`.
- Update `make_placement_agent` if present.

**Skip Deliverable 3 (the data-collection changes) for now** — we only need it for Part B.

## A2. Update the benchmark script for the heuristic strategy

**File**: `capstone_agent/Placement/benchmark_placement.py`

### A2a. Add CLI flags

```python
parser.add_argument(
    "--strategy", type=str, default="random",
    choices=["random", "model", "heuristic"],  # "value" will be added in Part B
    help="Placement strategy to test",
)
parser.add_argument(
    "--temperature", type=float, default=0.0,
    help="Temperature for heuristic sampling. 0 = deterministic argmax.",
)
parser.add_argument(
    "--board-seed", type=int, default=42,
    help="Seed for the fixed board layout",
)
parser.add_argument(
    "--game-seed-start", type=int, default=0,
    help="First game seed. Game i uses game_seed_start + i.",
)
parser.add_argument(
    "--results-log", type=str, default=None,
    help="Optional JSONL file to append benchmark results",
)
```

### A2b. Use the fixed board in every game

Check whether `benchmark_placement.py` already creates games with `catan_map=fixed_map`. If not, add:

```python
import random
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE

def _fixed_map(board_seed: int = 42):
    random.seed(board_seed)
    return CatanMap.from_template(BASE_MAP_TEMPLATE)
```

In `run_group`, construct each game as:
```python
fixed_map = _fixed_map(args.board_seed)
# In the loop:
game = Game(players=[blue, red], catan_map=fixed_map, seed=game_seed_start + game_idx)
```

This matches the training-data contract (fixed board, varying game seed). Without this, benchmark games play on different random boards and results aren't comparable to training.

### A2c. Hook in the heuristic strategy

When `args.strategy == "heuristic"`, instantiate `HeuristicPlacementAgent(temperature=args.temperature)` instead of going through `make_placement_agent` (unless `make_placement_agent` was updated to accept temperature).

### A2c-bis. CRITICAL: Fix HybridPlayer to pass `game=` and `playable_actions=`

**Bug in existing code**: `benchmark_placement.py`'s `HybridPlayer._placement_decide()` currently calls `placement_agent.select_action(obs, mask)` with NO keyword arguments. The heuristic agent (and the value agent in Part B) both require `game=` and `playable_actions=` to function. Without this fix, the benchmark will crash immediately.

Update `HybridPlayer._placement_decide`:

```python
def _placement_decide(self, game, playable_actions):
    opp_color = [c for c in game.state.colors if c != self.color][0]
    obs = np.array(
        get_capstone_observation(game, self.color, opp_color),
        dtype=np.float32,
    )
    mask = _make_action_mask(game)
    action_idx, _, _ = self.placement_agent.select_action(
        obs, mask,
        game=game,                      # NEW
        playable_actions=playable_actions,  # NEW
    )
    return capstone_to_action(action_idx, playable_actions)
```

This change is also required for the value-strategy benchmark in Part B.

### A2d. Log results to JSONL

After each group finishes, append a row to `--results-log` (if provided):
```python
from datetime import datetime, timezone

row = {
    "event": "benchmark_group",
    "label": strategy_label,
    "strategy": args.strategy,
    "temperature": args.temperature,
    "board_seed": args.board_seed,
    "game_seed_start": args.game_seed_start,
    "games": n,
    "wins": wins,
    "losses": losses,
    "draws": draws,
    "win_rate": wins / n if n else 0.0,
    "ci_lo": lo,
    "ci_hi": hi,
    "elapsed_seconds": elapsed,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
with open(args.results_log, "a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
```

## A3. Run the 500-game benchmark

```bash
python capstone_agent/Placement/benchmark_placement.py \
    --games 500 \
    --strategy heuristic \
    --temperature 0.0 \
    --board-seed 42 \
    --game-seed-start 500000 \
    --results-log capstone_agent/models/heuristic_benchmark_results.jsonl \
    --verbose
```

Notes:
- `--temperature 0.0` for benchmarking (deterministic argmax).
- `--game-seed-start 500000` is well above any seed used during training data collection (those used 0-44999 and 44000-93999), preventing accidental overlap.
- Keep `--board-seed 42` consistent with training data.
- The script should run both baseline (pure AlphaBeta vs AlphaBeta) and test (heuristic+AlphaBeta vs AlphaBeta) on the same seeds.

### A3-bis. Statistical significance via paired testing

Because both groups play the exact same 500 game seeds, this is a **paired comparison**, not two independent samples. Overlapping 95% CIs are NOT the right test. Use McNemar's test on the paired wins:

Inside the benchmark script (or as a post-processing step), record per-game outcomes for each group as two parallel lists `baseline_wins[i]` and `test_wins[i]`, each a 0/1 for game `i`. Then:

```python
from scipy.stats import binomtest

# b = games where baseline won but test lost
# c = games where test won but baseline lost
# (games where both won or both lost don't affect the test)
b = sum(1 for i in range(n) if baseline_wins[i] and not test_wins[i])
c = sum(1 for i in range(n) if test_wins[i] and not baseline_wins[i])
discordant = b + c

# Exact binomial McNemar's test
if discordant > 0:
    result = binomtest(c, discordant, p=0.5, alternative="greater")
    p_value = result.pvalue
    print(f"McNemar exact test: b={b}, c={c}, p={p_value:.4f}")
    print(f"Test strategy wins {c}/{discordant} of discordant pairs")
```

Also append paired results to `--results-log` with `baseline_only_wins`, `test_only_wins`, `both_won`, `neither_won`, and `mcnemar_pvalue` so later analyses can re-run the test.

A p-value < 0.05 with `c > b` means the heuristic is a statistically significant improvement over pure AlphaBeta. Without this, a win rate of 52% could be noise.

## A4. Verification for Part A

1. Compile: `python -m py_compile capstone_agent/Placement/placement_heuristic.py capstone_agent/Placement/PlacementAgent.py capstone_agent/Placement/benchmark_placement.py`
2. Heuristic sanity check (from the existing plan's Test 1).
3. Tiny 10-game smoke benchmark:
   ```bash
   python capstone_agent/Placement/benchmark_placement.py \
       --games 10 --strategy heuristic --temperature 0.0 --skip-baseline
   ```
4. Full 500-game run (A3).

### Decision point after Part A

- **Win rate ≥ 55%**: Heuristic is meaningfully strong. Proceed with Part B.
- **Win rate 50-55%**: Marginal. Proceed with Part B but expect small gains.
- **Win rate ≈ 50% or below**: The heuristic is not actually helping. Diagnose the scoring before continuing.

---

# PART B — Value Function + 1-Ply Search

## B1. Concept

Train `V(state) → P(acting player wins)`. At decision time, for each legal placement, hypothetically execute it, extract the resulting observation, feed it to V, and pick the action with the highest predicted win probability.

**Why this can exceed the heuristic**: the heuristic is hand-coded rules. V learns from actual game outcomes, so it can discover patterns the heuristic misses — e.g., specific node pairs that synergize, or port-heavy setups the heuristic under-rates.

**State**: the 850-dim compact placement observation (same features as the policy model).
**Label**: 1.0 if acting player won the game, else 0.0.
**Loss**: binary cross-entropy (BCE) with logits.

Each game yields 8 (state, label) examples — one per opening ply, matching `OPENING_STEP_COUNT = 8` in `placement_supervised_dataset.py`. The accumulator records the 8 opening actions; `iter_reconstructed_examples` yields the pre-action state alongside each action. We keep the same state sequence, but redirect the training target from "action taken" to "did this actor win."

## B2. Data collection for Part B

Implement the heuristic-placement data collection updates from `CODEX_PLAN_HEURISTIC_PLACEMENT.md` Deliverable 3 (the changes to `collect_compact_placement_data.py`). Then collect 50K games with:

```bash
python capstone_agent/Placement/collect_compact_placement_data.py \
    --games 50000 \
    --games-per-chunk 5000 \
    --out-dir capstone_agent/data/heuristic_placement_50k \
    --strategy heuristic \
    --temperature 0.5 \
    --fixed-board --board-seed 42 \
    --seed 200000
```

Temperature = 0.5 during collection gives diverse states. (Lower = less variety, so V doesn't see enough state-space. Higher = too random, back to noise.)

## B3. Value network architecture

**New file**: `capstone_agent/Placement/PlacementValueModel.py`

```python
"""Value network for opening placement: state -> win probability (as logit)."""

import torch
import torch.nn as nn

try:
    from .placement_features import COMPACT_PLACEMENT_FEATURE_SIZE
except ImportError:
    from placement_features import COMPACT_PLACEMENT_FEATURE_SIZE


class PlacementValueModel(nn.Module):
    """850-d placement observation -> scalar logit for P(acting player wins)."""

    def __init__(
        self,
        obs_size: int = COMPACT_PLACEMENT_FEATURE_SIZE,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)  # (batch,) logit
```

## B4. Value function training script

**New file**: `capstone_agent/Placement/train_placement_value.py`

Mirror the structure of `train_compact_placement_supervised.py`, with these differences:

1. **Label**: `label = float(example.actor_seat == example.winner_seat)`. Read `actor_seat` and `winner_seat` from the `ReconstructedPlacementExample` dataclass.
2. **Loss**: `torch.nn.BCEWithLogitsLoss(reduction='none')(logits, labels)`, then multiply by the example weight and take the mean.
3. **Weights**: Default to `outcome_weighted` with `win_weight=1.0, loss_weight=1.0` — for value learning we want both classes contributing equally. Exposing the flags lets you tune it.
4. **Drop prompt masking**: the value function doesn't need per-prompt legal-action masks. Just feed the observation `x` to the model.
5. **Metrics per epoch**:
   - `train_loss`, `val_loss` (BCE mean)
   - `val_accuracy`: `mean((sigmoid(logit) > 0.5) == label)`
   - `val_auroc`: `sklearn.metrics.roc_auc_score(labels, sigmoid(logits))` — more meaningful than accuracy for this task because label imbalance and noise make 50%-accurate a meaningful baseline
   - `val_brier`: `mean((sigmoid(logit) - label) ** 2)` — calibration quality
6. **Reuse everything else**: per-game train/val split, optimizer state persistence, `--resume`, CosineAnnealingLR, early stopping, MPS support, `torch.compile()`.

CLI:
```bash
python capstone_agent/Placement/train_placement_value.py \
    --data capstone_agent/data/heuristic_placement_50k \
    --out capstone_agent/models/placement_value.pt \
    --epochs 50 --batch-size 512 --lr 3e-4 \
    --selection-mode outcome_weighted \
    --win-weight 1.0 --loss-weight 1.0 \
    --patience 10 \
    --metrics-log capstone_agent/models/placement_value_metrics.jsonl
```

**Expected behavior**: val_auroc should climb above 0.55 within a few epochs. If it stays at 0.5, the value function isn't learning. Val loss around 0.69 means "predicting 50/50 for every state" — a sign of underfitting.

## B5. Value-guided placement agent

**Required imports** (make sure these are all resolved — the snippet uses lazy imports inside the class body but list them here for clarity):
- `get_compact_placement_observation` from `capstone_agent/Placement/placement_features.py`
- `catanatron_action_to_capstone_index` from `catanatron/catanatron/gym/envs/action_translator.py`
- `PlacementValueModel` from the file created in B3
- `get_device` from `capstone_agent/device.py`

**File**: `capstone_agent/Placement/PlacementAgent.py` (add a new class)

```python
class ValueSearchPlacementAgent:
    """1-ply search using a value network.

    For each legal action: copy the game, execute the action, extract the
    resulting compact placement observation, evaluate V, and pick the
    action with the highest predicted win probability.
    """

    def __init__(self, value_model_path: str, hidden_size: int = 256, **_kwargs):
        import torch
        try:
            from .PlacementValueModel import PlacementValueModel
            from ..device import get_device
        except ImportError:
            from PlacementValueModel import PlacementValueModel
            from device import get_device

        self.device = get_device()
        self.model = PlacementValueModel(hidden_size=hidden_size).to(self.device)
        state = torch.load(value_model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def select_action(self, state, mask, **kwargs):
        import torch
        import numpy as np

        game = kwargs.get("game")
        playable_actions = kwargs.get("playable_actions")
        if game is None or playable_actions is None:
            raise ValueError(
                "ValueSearchPlacementAgent requires game= and playable_actions= kwargs"
            )

        try:
            from .placement_features import get_compact_placement_observation
        except ImportError:
            from placement_features import get_compact_placement_observation
        from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index

        current_color = game.state.colors[game.state.current_player_index]

        candidate_obs = []
        candidate_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)
            next_obs = get_compact_placement_observation(game_copy, current_color)
            candidate_obs.append(next_obs)
            candidate_actions.append(action)

        batch = torch.as_tensor(
            np.stack(candidate_obs), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()

        best_idx = int(np.argmax(probs))
        chosen_action = candidate_actions[best_idx]
        capstone_idx = catanatron_action_to_capstone_index(chosen_action)
        return (capstone_idx, 0.0, float(probs[best_idx]))

    def store(self, *args, **kwargs): pass
    def train(self, *args, **kwargs): pass
    def load(self, path): pass  # loaded in __init__
    def save(self, *args, **kwargs): pass
```

**Performance caveat (read before running the full benchmark)**: `game.copy()` in catanatron does a deep copy including `Board.copy()`, which duplicates connected-components caches and other state. This is nontrivial — expect it to take several milliseconds per call.

Cost estimate: a settlement decision has up to ~30 legal options early and fewer later. Each of the 8 opening plies runs one call to `select_action`. So each game does roughly `sum(legal_options at each ply)` copies — ballpark 100-200 copies per game. At 5ms per copy, that's ~0.5-1.0 seconds of overhead per game JUST for 1-ply search, on top of the AlphaBeta main-game work. For 500 games, that's 5-10 extra minutes. Probably tolerable, but verify by timing the 10-game smoke test.

**Mitigation if too slow**: instead of `game.copy() + execute()`, compute the next-state observation manually by:
1. Starting from the current observation (already available as `state`).
2. Setting the appropriate occupancy bit (settlement node or road edge) to `1.0`.
3. Updating the 4 step-count indicators at the tail.
4. Feeding that directly to V.

This avoids all game-copy overhead. Defer this optimization unless the 10-game smoke test shows > 1 second per game.

**Important**: benchmark this before running the full 500-game test. If each game takes > 30 seconds end-to-end, something is wrong — abort and investigate.

## B6. Benchmark updates for value agent

**File**: `capstone_agent/Placement/benchmark_placement.py`

Extend the strategy choices:
```python
parser.add_argument(
    "--strategy", type=str, default="random",
    choices=["random", "model", "heuristic", "value"],
)
parser.add_argument(
    "--value-model", type=str, default=None,
    help="Path to trained value model (required for --strategy value)",
)
```

When `args.strategy == "value"`:
```python
if args.value_model is None:
    parser.error("--value-model is required when --strategy is 'value'")
placement_agent = ValueSearchPlacementAgent(value_model_path=args.value_model)
```

Run the 500-game benchmark:
```bash
python capstone_agent/Placement/benchmark_placement.py \
    --games 500 \
    --strategy value \
    --value-model capstone_agent/models/placement_value.pt \
    --board-seed 42 \
    --game-seed-start 500000 \
    --results-log capstone_agent/models/value_benchmark_results.jsonl \
    --verbose
```

**Critical**: use the **same `--game-seed-start` as the heuristic benchmark** so both strategies face the identical 500 games. This is a paired comparison, which cuts variance dramatically.

## B7. Verification for Part B

1. Compile: `python -m py_compile` on all new/modified files.
2. Value model forward pass:
   ```bash
   python3 -c "
   import torch
   from capstone_agent.Placement.PlacementValueModel import PlacementValueModel
   m = PlacementValueModel()
   x = torch.randn(4, 850)
   out = m(x)
   assert out.shape == (4,)
   print('Value model OK')
   "
   ```
3. Training smoke test (reuses existing data to save time):
   ```bash
   python capstone_agent/Placement/train_placement_value.py \
       --data capstone_agent/data/apr14_10am_run1 \
       --out /tmp/v_smoke.pt --epochs 3 --batch-size 64 \
       --metrics-log /tmp/v_smoke_metrics.jsonl
   ```
   Loss should decrease; val_auroc should be > 0.5.
4. Tiny 10-game value benchmark:
   ```bash
   python capstone_agent/Placement/benchmark_placement.py \
       --games 10 --strategy value --value-model /tmp/v_smoke.pt \
       --skip-baseline
   ```
5. Full pipeline: 50K heuristic data collection → train V → 500-game value benchmark.

---

## Summary of files touched

### Part A
- **From `CODEX_PLAN_HEURISTIC_PLACEMENT.md`**: create `placement_heuristic.py`, add `HeuristicPlacementAgent`.
- **New in this plan**: benchmark script updates (fixed board, heuristic strategy, results log).

### Part B
- **From `CODEX_PLAN_HEURISTIC_PLACEMENT.md` Deliverable 3**: data collection changes (if not already done).
- **New**: `PlacementValueModel.py`, `train_placement_value.py`, `ValueSearchPlacementAgent` class, value-strategy benchmark flag.

## Execution order

1. Implement Part A (follow `CODEX_PLAN_HEURISTIC_PLACEMENT.md` Deliverables 1, 2, 4 + this plan A2).
2. Run the 500-game heuristic benchmark (A3).
3. **Report back to the user with the win rate AND the McNemar p-value from A3-bis.** Decide whether to proceed.
4. Implement Part B (B2–B6).
5. Run the 500-game value benchmark (B6).
6. Compare heuristic vs value win rates on the same seeds — again with McNemar's test, not independent CIs.

## Known Risks (surfaced by code-review)

These are acknowledged risks, not bugs. The plan proceeds anyway because the alternative is more scope. Document them so we don't pretend they aren't there.

1. **Heuristic is strategy-level crude.** No opponent blocking, no longest-road reasoning, no step-aware weighting (1st vs 2nd settlement use identical weights). These are v2 improvements. Iterate if the 500-game benchmark shows the heuristic barely beats AlphaBeta.

2. **V may not surpass the heuristic.** V trained on heuristic+temp data learns the heuristic's state distribution plus noisy outcomes. Temperature (0.5) is our only source of counterfactual variation. If V underperforms the heuristic in the 500-game benchmark, the experiment still teaches us something — don't retry endlessly.

3. **Distribution shift at deployment.** V is trained against heuristic-temp opponents but deployed against deterministic AlphaBeta. Ranking should still hold even if calibration is off; verify by inspecting the value histogram on benchmark games.

4. **Fixed board seed = 42.** Everything is trained and evaluated on one board layout. V will not transfer to random boards. For this capstone project, that's acceptable — but document it.

5. **Draw / timeout bias.** `Game.play()` returns `None` at `TURNS_LIMIT`. The compact-data pipeline skips `no_winner` games. If one placement strategy causes systematically more timeouts than another (e.g., because it builds weaker early positions that can't reach 10 VP), the filter is biased. Add a telemetry line to both data collection and benchmark output: total games, games with winner, games that timed out, draw rate. If draw rate > 5%, investigate.

6. **50K games × 8 plies = 400K examples for V.** Plenty of samples, but labels are noisy. BCE may converge to "predict the seat/tempo baseline" rather than "predict the effect of this specific placement." Val AUROC should creep above 0.55 by epoch 3; if it stays at 0.50, V isn't learning anything useful from the opening state.
