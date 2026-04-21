## The Placement algorithm in plain English

For each placement decision (e.g., "where should I put my first settlement?"), it:

1. **For each legal action**, simulates playing it and then simulates the rest of the opening phase
2. Evaluates the final post-opening board position using a scoring function
3. Picks the action that leads to the best final position

It's essentially **minimax search through the entire opening phase** — not a 1-ply lookahead, and not the placement heuristic we designed.

## The step-by-step mechanics

### Step 1: For each candidate action, run a rollout
`select_action()` at line 701 iterates through every legal action. For each one, it calls `_completed_opening_value()` (line 732) which:

```python
def _completed_opening_value(self, game, action, root_color, value_fn):
    rollout = self._complete_opening(game, action, root_color, value_fn)
    return value_fn(rollout, root_color)
```

### Step 2: Complete the opening phase by greedy minimax
`_complete_opening()` (line 374) copies the game, plays the candidate action, then loops until the opening is over:

```python
while rollout.state.is_initial_build_phase and rollout.winning_color() is None:
    rollout_action = self._best_rollout_action(rollout, root_color, value_fn)
    rollout.execute(rollout_action)
```

This is the important part: it simulates the **remaining placements of BOTH players**. On each step inside the rollout:

- If it's the root player's turn → pick the action that **maximizes** the value function
- If it's the opponent's turn → pick the action that **minimizes** the value function

That's `_best_rollout_action()` at line 678 (the "stable" override):

```python
def _best_rollout_action(self, game, root_color, value_fn):
    maximizing = game.state.current_color() == root_color
    best_value = float("-inf") if maximizing else float("inf")
    for action_idx, action in self._indexed_actions(game.playable_actions):
        game_copy = game.copy()
        game_copy.execute(action)
        value = value_fn(game_copy, root_color)
        if self._better_value(value, best_value, action_idx, best_action_idx, maximizing):
            # ...
```

So it's a 1-ply greedy minimax **at every level** of the opening — for each of the 8 plies remaining, it looks one action ahead and picks the best under the assumption the other player is also playing greedily against our value function.

### Step 3: Evaluate the final position
Once all 8 opening plies are complete (and the opening is over), the rollout stops. `value_fn(rollout, root_color)` scores the resulting board from the root player's perspective. That scalar is what gets assigned to the original candidate action.

### Step 4: Pick the best
After evaluating every legal action, pick the one with the highest value. The **"Stable"** part is line 664's `_better_value()` function: when two actions produce nearly identical values (within `1e-9`), it breaks ties by picking the lower capstone action index. This makes decisions reproducible across runs.

## What value function is it using?

Not the placement heuristic we designed. Look at `_value_fn()` line 271:

```python
def _value_fn(self):
    from catanatron.players.value import get_value_fn
    return get_value_fn(self.value_fn_builder_name or "base_fn", self.params)
```

It uses catanatron's existing `base_fn` — the same hand-crafted value function AlphaBeta uses. It has 13 weighted factors: `public_vps`, `production`, `enemy_production`, `reachable_production_1`, `buildable_nodes`, `longest_road`, `hand_synergy`, etc. You can see it in `catanatron/catanatron/players/value.py`.

## So why does this beat pure AlphaBeta?

This is subtle and actually quite clever. Both agents use the same `base_fn` value function. The difference is **what they search over**:

- **Pure AlphaBeta**: fixed search depth (2) applied to every decision including placements. During placement, it only looks 2 plies deep, which isn't enough to see the consequences of a placement decision.
- **StableRolloutValue**: specifically for placement decisions, it searches the **entire remaining opening phase** (up to 7 plies for the very first move). This is much deeper than depth-2 AlphaBeta's lookahead during placement.

By searching deeper than AlphaBeta can afford to during the opening, it makes better opening decisions. Then it hands off to AlphaBeta for the rest of the game.

## The cost

Roughly: 30 legal actions × 8 plies of rollout × 30 legal actions per ply × `game.copy()` cost per evaluation ≈ **~7,000 game copies per opening decision**. That's why the 500-game benchmark took ~15-21 minutes for the test group — each game is doing a lot of rollout work just in the first 8 moves.

## What it's NOT

- ❌ Not using `placement_heuristic.py` at all (that file became dead code for this winner)
- ❌ Not using a neural network
- ❌ Not using the 1-ply search from the original Plan B
- ❌ Not using the heuristic's multi-factor scoring (weighted production, port synergy, etc.)

Codex essentially looked at the plan, decided the right thing was to **go deeper in search at the decision point where it matters (placement) using catanatron's existing value function**, and skipped the neural-net detour entirely. That was a good call — it got to statistically significant results much faster than training a value network would have.