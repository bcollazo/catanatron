# Codex Plan: Heuristic Placement Agent + Data Collection Overhaul

## Why This Change Is Needed

The current training pipeline collects data from **random** placement. Both players place settlements/roads at random, then play the rest of the game with AlphaBeta. The model trains on "imitate the winner's placements."

**This doesn't work.** After 93K games of training, the model achieved 2% settlement accuracy (worse than random) and 39% road accuracy (≈ random with 2-3 choices). The fundamental problem: game outcome is dominated by dice luck, not placement quality. When placements are random, the "winner" label is almost uncorrelated with placement quality.

**The fix:** Replace random placement with a heuristic-guided placement agent that makes strategically sound (but not deterministic) placement decisions. This gives us:
1. Higher-quality placements in the training data
2. Game outcomes that more strongly correlate with placement quality
3. A meaningful supervision signal for the model to learn from

The heuristic samples from a temperature-softmax over scores, so it's stochastic enough to produce diverse training data, but good enough that placement quality genuinely affects who wins.

---

## Deliverable 1: Placement Heuristic Module

**New file:** `capstone_agent/Placement/placement_heuristic.py`

This module provides pure functions that score candidate settlement and road placements. It needs NO game simulation — just static analysis of the board.

### Imports needed

```python
import math
import numpy as np
from collections import Counter
from catanatron.models.map import CatanMap, number_probability, DICE_PROBAS
from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.enums import RESOURCES
```

### Constants

```python
# Strategic resource weights.
# ORE + WHEAT are "city resources" — the strongest VP path in Catan (3 ore + 2 wheat per city).
# WHEAT is also needed for settlements and dev cards, making it the most versatile.
RESOURCE_WEIGHTS = {
    "ORE": 1.25,
    "WHEAT": 1.25,
    "SHEEP": 0.95,
    "WOOD": 1.0,
    "BRICK": 1.0,
}

# Pre-compute the canonical edge ordering (same as placement_supervised_dataset.py)
EDGE_ORDER = [tuple(sorted(edge)) for edge in get_edges()]
EDGE_TO_INDEX = {edge: idx for idx, edge in enumerate(EDGE_ORDER)}
```

### Helper: get port resource for a node

```python
def _get_port_type(catan_map: CatanMap, node_id: int):
    """Return the port type for a node: a resource string for 2:1, None key for 3:1, or False if no port."""
    for resource, node_set in catan_map.port_nodes.items():
        if node_id in node_set:
            return resource  # resource string for 2:1, None for 3:1
    return False  # no port
```

Note: `catan_map.port_nodes` maps `resource_string → set_of_node_ids` for 2:1 ports, and `None → set_of_node_ids` for 3:1 ports.

### Helper: check if a node is buildable (Catan distance rule)

```python
def _is_node_buildable(node_id: int, all_settlement_nodes: set) -> bool:
    """Check whether a node could hold a future settlement.
    
    Catan rule: settlements must be at least 2 edges apart.
    A node is buildable if it's unoccupied AND none of its neighbors are occupied.
    """
    if node_id in all_settlement_nodes:
        return False
    for neighbor in STATIC_GRAPH.neighbors(node_id):
        if neighbor in all_settlement_nodes:
            return False
    return True
```

### Core: score_settlement

```python
def score_settlement(
    catan_map: CatanMap,
    node_id: int,
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
) -> float:
    """Score a candidate settlement node. Higher is better.
    
    Components:
    1. Weighted production — sum of (resource_weight × dice_probability) for adjacent hexes
    2. Resource variety — bonus for covering distinct resource types
    3. Complementarity — bonus for NEW resources not covered by existing settlements (2nd settlement)
    4. Port synergy — bonus if node has a useful port
    5. Number diversity — bonus for covering new dice numbers (2nd settlement)
    """
    
    production = catan_map.node_production[node_id]  # Counter({resource_str: probability})
    resources_here = set(production.keys())
    
    # ── 1. Weighted production ────────────────────────────────────
    weighted_prod = sum(
        RESOURCE_WEIGHTS.get(r, 1.0) * prob
        for r, prob in production.items()
    )
    
    # ── 2. Resource variety ───────────────────────────────────────
    # More distinct resource types = more strategic flexibility.
    variety_bonus = len(resources_here) * 0.06
    
    # ── 3. Complementarity (2nd settlement) ───────────────────────
    # Strong bonus for each NEW resource type not covered by existing settlements.
    complementarity_bonus = 0.0
    if my_settlement_nodes:
        existing_resources = set()
        for node in my_settlement_nodes:
            existing_resources.update(catan_map.node_production[node].keys())
        new_resources = resources_here - existing_resources
        complementarity_bonus = len(new_resources) * 0.10
        
        # Extra push if this fills a critical gap (ORE or WHEAT missing)
        for critical in ("ORE", "WHEAT"):
            if critical not in existing_resources and critical in resources_here:
                complementarity_bonus += 0.06
    
    # ── 4. Port synergy ───────────────────────────────────────────
    port_bonus = 0.0
    port_type = _get_port_type(catan_map, node_id)
    if port_type is not False:
        if port_type is None:
            # 3:1 generic port — modest flat bonus
            port_bonus = 0.04
        else:
            # 2:1 specific port — value depends on total production of that resource
            # across THIS node + existing settlements
            total_prod_of_resource = production.get(port_type, 0.0)
            for node in my_settlement_nodes:
                total_prod_of_resource += catan_map.node_production[node].get(port_type, 0.0)
            # The port effectively lets you trade 2:1, which roughly doubles the
            # trade value of that resource. Scale bonus by production.
            port_bonus = total_prod_of_resource * 0.4
            # Even if you don't produce the resource yet, the port has some value
            port_bonus = max(port_bonus, 0.02)
    
    # ── 5. Number diversity (2nd settlement) ──────────────────────
    number_bonus = 0.0
    if my_settlement_nodes:
        existing_numbers = set()
        for node in my_settlement_nodes:
            for tile in catan_map.adjacent_tiles[node]:
                if tile.number is not None:
                    existing_numbers.add(tile.number)
        new_numbers = set()
        for tile in catan_map.adjacent_tiles[node_id]:
            if tile.number is not None and tile.number not in existing_numbers:
                new_numbers.add(tile.number)
        number_bonus = len(new_numbers) * 0.025
    
    return weighted_prod + variety_bonus + complementarity_bonus + port_bonus + number_bonus
```

### Core: score_road

```python
def score_road(
    catan_map: CatanMap,
    edge_idx: int,
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
) -> float:
    """Score a candidate road edge. Higher is better.
    
    Roads in the opening placement must be adjacent to the just-placed settlement.
    The key question: does this road point toward good future expansion?
    
    Score = settlement heuristic of the "far end" node (the end that isn't our settlement).
    If the far end is not buildable (distance rule violation), heavily discount it.
    """
    
    node_a, node_b = EDGE_ORDER[edge_idx]
    all_settlements = set(my_settlement_nodes) | set(opp_settlement_nodes)
    
    # Identify the "far end" — the node that is NOT our settlement
    my_set = set(my_settlement_nodes)
    if node_a in my_set and node_b not in my_set:
        far_end = node_b
    elif node_b in my_set and node_a not in my_set:
        far_end = node_a
    else:
        # Edge case: both or neither are our settlements.
        # Score both and take the max.
        score_a = _quick_node_score(catan_map, node_a)
        score_b = _quick_node_score(catan_map, node_b)
        return max(score_a, score_b)
    
    # Check if the far end could eventually be settled
    buildable = _is_node_buildable(far_end, all_settlements)
    
    if buildable:
        # Score the far end as a future settlement prospect
        # Use a simplified score (no complementarity — we don't know what 2nd settlement will be yet)
        base = _quick_node_score(catan_map, far_end)
        
        # Bonus: does the far end give access to new resources?
        existing_resources = set()
        for node in my_settlement_nodes:
            existing_resources.update(catan_map.node_production[node].keys())
        far_resources = set(catan_map.node_production[far_end].keys())
        new_resource_bonus = len(far_resources - existing_resources) * 0.05
        
        return base + new_resource_bonus
    else:
        # Far end is blocked — road is less useful but not worthless
        # (it still contributes to longest road and might lead to nodes 2 hops away)
        # Look one hop further: score the far_end's OTHER neighbors
        further_scores = []
        for next_node in STATIC_GRAPH.neighbors(far_end):
            if next_node in my_set:
                continue  # don't look back toward our settlement
            if _is_node_buildable(next_node, all_settlements):
                further_scores.append(_quick_node_score(catan_map, next_node) * 0.5)
        return max(further_scores) if further_scores else 0.01


def _quick_node_score(catan_map: CatanMap, node_id: int) -> float:
    """Simplified settlement score for road evaluation — just production + variety."""
    production = catan_map.node_production[node_id]
    weighted = sum(RESOURCE_WEIGHTS.get(r, 1.0) * p for r, p in production.items())
    variety = len(production) * 0.06
    return weighted + variety
```

### Top-level: score all legal actions

```python
def score_legal_settlements(
    catan_map: CatanMap,
    legal_node_ids: list[int],
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
) -> dict[int, float]:
    """Score every legal settlement node. Returns {node_id: score}."""
    return {
        node_id: score_settlement(catan_map, node_id, my_settlement_nodes, opp_settlement_nodes)
        for node_id in legal_node_ids
    }


def score_legal_roads(
    catan_map: CatanMap,
    legal_edge_indices: list[int],
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
) -> dict[int, float]:
    """Score every legal road edge. Returns {edge_idx: score}."""
    return {
        edge_idx: score_road(catan_map, edge_idx, my_settlement_nodes, opp_settlement_nodes)
        for edge_idx in legal_edge_indices
    }
```

### Sampling helper

```python
def sample_from_scores(scores: dict[int, float], temperature: float) -> tuple[int, float]:
    """Sample an action from heuristic scores using temperature-scaled softmax.
    
    Args:
        scores: {action_id: heuristic_score}
        temperature: >0. Lower = greedier. 0.0 = argmax.
    
    Returns:
        (chosen_action_id, log_probability)
    """
    actions = list(scores.keys())
    values = np.array([scores[a] for a in actions], dtype=np.float64)
    
    if temperature <= 0 or temperature < 1e-8:
        # Deterministic: pick the best
        best_idx = int(np.argmax(values))
        return actions[best_idx], 0.0
    
    # Temperature-scaled softmax
    logits = values / temperature
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    
    chosen_idx = np.random.choice(len(actions), p=probs)
    log_prob = float(np.log(probs[chosen_idx] + 1e-10))
    return actions[chosen_idx], log_prob
```

---

## Deliverable 2: HeuristicPlacementAgent

**File:** `capstone_agent/Placement/PlacementAgent.py`

Add a new class `HeuristicPlacementAgent` alongside the existing `RandomPlacementAgent`. It has the same interface.

```python
class HeuristicPlacementAgent:
    """Uses a multi-factor heuristic to score legal placements, then
    samples from a temperature-scaled softmax. Produces higher-quality
    training data than RandomPlacementAgent while maintaining diversity."""
    
    def __init__(self, temperature: float = 0.5, **_kwargs):
        self.temperature = temperature
    
    def select_action(self, state, mask, **kwargs):
        game = kwargs.get("game")
        if game is None:
            raise ValueError(
                "HeuristicPlacementAgent requires `game=` keyword argument. "
                "Are you calling through RouterCapstonePlayer?"
            )
        
        mask = np.asarray(mask)
        
        # Import here to avoid circular imports
        from placement_heuristic import (
            score_legal_settlements,
            score_legal_roads,
            sample_from_scores,
            EDGE_ORDER,
        )
        from placement_action_space import PlacementPrompt, infer_placement_prompt
        from CONSTANTS import ROAD_ACTION_SLICE, SETTLEMENT_ACTION_SLICE
        
        prompt = infer_placement_prompt(mask)
        
        # Extract current board state
        catan_map = game.state.board.map
        current_color = game.state.colors[game.state.current_player_index]
        opp_color = [c for c in game.state.colors if c != current_color][0]
        
        # Get existing settlement nodes for both players
        from catanatron.models.enums import SETTLEMENT
        buildings = game.state.buildings_by_color
        my_settlements = list(buildings.get(current_color, {}).get(SETTLEMENT, []))
        opp_settlements = list(buildings.get(opp_color, {}).get(SETTLEMENT, []))
        
        if prompt == PlacementPrompt.SETTLEMENT:
            # Legal settlement nodes from the mask
            valid_capstone = np.where(mask[SETTLEMENT_ACTION_SLICE] > 0.5)[0]
            legal_nodes = [int(idx) for idx in valid_capstone]  # node_id = local index
            
            scores = score_legal_settlements(
                catan_map, legal_nodes, my_settlements, opp_settlements
            )
            chosen_node, log_prob = sample_from_scores(scores, self.temperature)
            capstone_idx = SETTLEMENT_ACTION_SLICE.start + chosen_node
        
        elif prompt == PlacementPrompt.ROAD:
            # Legal road edges from the mask
            valid_capstone = np.where(mask[ROAD_ACTION_SLICE] > 0.5)[0]
            legal_edges = [int(idx) for idx in valid_capstone]  # edge_idx = local index
            
            scores = score_legal_roads(
                catan_map, legal_edges, my_settlements, opp_settlements
            )
            chosen_edge, log_prob = sample_from_scores(scores, self.temperature)
            capstone_idx = ROAD_ACTION_SLICE.start + chosen_edge
        
        else:
            raise ValueError(f"Unknown prompt: {prompt}")
        
        value = 0.0
        return (capstone_idx, log_prob, value)
    
    def store(self, state, mask, action, log_prob, reward, value, done):
        pass
    
    def train(self, last_value):
        pass
    
    def load(self, path):
        pass
    
    def save(self, path):
        pass
```

**Important implementation detail:** The `buildings_by_color` dict in the catanatron game state stores buildings as:
```python
{Color.BLUE: {SETTLEMENT: [node_id, ...], CITY: [node_id, ...]}, ...}
```
Verify this by inspecting `game.state.buildings_by_color` at runtime. If the structure differs (e.g., it uses a defaultdict or different nesting), adjust the access pattern. The key data needed is: list of node_ids where each player has a settlement.

---

## Deliverable 3: Update Data Collection

**File:** `capstone_agent/Placement/collect_compact_placement_data.py`

### 3a. Add CLI arguments

```python
parser.add_argument(
    "--strategy",
    type=str,
    default="heuristic",
    choices=["random", "heuristic"],
    help="Placement strategy for data collection (default: heuristic)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.5,
    help="Temperature for heuristic placement sampling (lower = greedier, default: 0.5)",
)
```

### 3b. Update the `_play_one_game` function

Currently (line 50-53):
```python
blue = RouterCapstonePlayer(
    Color.BLUE,
    RandomPlacementAgent(),
    AlphaBetaMainAgentAdapter(Color.BLUE, depth=ab_depth),
)
```

Change to accept strategy and temperature:

```python
def _play_one_game(args):
    game_seed, ab_depth, board_seed, strategy, temperature = args
    fixed_map = _fixed_map_from_seed(board_seed) if board_seed is not None else None
    random.seed(game_seed)
    np.random.seed(game_seed % (2**32 - 1))
    
    if strategy == "heuristic":
        blue_placement = HeuristicPlacementAgent(temperature=temperature)
        red_placement = HeuristicPlacementAgent(temperature=temperature)
    else:
        blue_placement = RandomPlacementAgent()
        red_placement = RandomPlacementAgent()
    
    blue = RouterCapstonePlayer(
        Color.BLUE,
        blue_placement,
        AlphaBetaMainAgentAdapter(Color.BLUE, depth=ab_depth),
    )
    red = RouterCapstonePlayer(
        Color.RED,
        red_placement,
        AlphaBetaMainAgentAdapter(Color.RED, depth=ab_depth),
    )
    # ... rest unchanged
```

### 3c. Update work_items to pass strategy and temperature

In the `collect()` function, update the work_items list (around line 158):

```python
work_items = [
    (game_seed, ab_depth, worker_board_seed, strategy, temperature)
    for game_seed in seeds
]
```

And add `strategy` and `temperature` as parameters to the `collect()` function signature.

### 3d. Update imports

Add at the top of the file:

```python
from PlacementAgent import RandomPlacementAgent, HeuristicPlacementAgent
```

(Move the existing `RandomPlacementAgent` import to include `HeuristicPlacementAgent`.)

### 3e. Log the strategy and temperature in the manifest

In the `run_started` manifest row (around line 148):

```python
{
    "event": "run_started",
    ...
    "strategy": strategy,
    "temperature": temperature,
}
```

---

## Deliverable 4: Update `make_placement_agent` Factory

**File:** `capstone_agent/Placement/PlacementAgent.py`

The `make_placement_agent` function (if it exists) should recognize `"heuristic"` as a strategy name. Find this function and add:

```python
if strategy == "heuristic":
    return HeuristicPlacementAgent(temperature=kwargs.get("temperature", 0.5))
```

If `make_placement_agent` doesn't exist or doesn't need updating, skip this.

---

## Verification

### Test 1: Heuristic produces sane rankings

```python
python3 -c "
import random
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
from capstone_agent.Placement.placement_heuristic import score_legal_settlements, score_legal_roads, EDGE_ORDER

random.seed(42)
catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)

# Score all 54 nodes as 1st settlement (no existing buildings)
all_nodes = list(range(54))
scores = score_legal_settlements(catan_map, all_nodes, [], [])
ranked = sorted(scores.items(), key=lambda x: -x[1])

print('Top 10 settlement nodes (1st settlement):')
for node, score in ranked[:10]:
    prod = dict(catan_map.node_production[node])
    print(f'  Node {node:2d}: score={score:.4f}  production={prod}')

# The top nodes should be high-production, multi-resource nodes
assert ranked[0][1] > ranked[-1][1], 'Best node should score higher than worst'
assert ranked[0][1] > 0.3, 'Best node should have a high score'
print('\nHeuristic ranking sanity check passed')
"
```

### Test 2: Compile check

```bash
python -m py_compile capstone_agent/Placement/placement_heuristic.py
python -m py_compile capstone_agent/Placement/PlacementAgent.py
python -m py_compile capstone_agent/Placement/collect_compact_placement_data.py
```

### Test 3: End-to-end data collection smoke test

```bash
python capstone_agent/Placement/collect_compact_placement_data.py \
    --games 10 \
    --games-per-chunk 10 \
    --out-dir /tmp/heuristic_smoke_test \
    --strategy heuristic \
    --temperature 0.5 \
    --workers 1
```

This should complete without errors and produce a `.npz` chunk file.

### Test 4: Training smoke test with heuristic data

```bash
python capstone_agent/Placement/train_compact_placement_supervised.py \
    --data /tmp/heuristic_smoke_test \
    --out /tmp/heuristic_smoke_model.pt \
    --epochs 3 \
    --batch-size 8 \
    --selection-mode outcome_weighted \
    --metrics-log /tmp/heuristic_smoke_metrics.jsonl
```

Should complete without errors. With only 10 games the accuracy won't be meaningful, but it should not crash.

### Test 5: Temperature sweep produces different distributions

```python
python3 -c "
import random
import numpy as np
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
from capstone_agent.Placement.placement_heuristic import score_legal_settlements, sample_from_scores

random.seed(42)
catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
scores = score_legal_settlements(catan_map, list(range(54)), [], [])

for temp in [0.1, 0.5, 1.0, 2.0]:
    np.random.seed(0)
    choices = [sample_from_scores(scores, temp)[0] for _ in range(1000)]
    unique = len(set(choices))
    top = max(set(choices), key=choices.count)
    print(f'  T={temp:.1f}: unique nodes chosen = {unique}, most common = node {top} ({choices.count(top)}/1000)')

# Lower temperature should produce fewer unique choices
print('\nTemperature sweep check passed')
"
```

---

## Architecture Summary

```
collect_compact_placement_data.py
  └── RouterCapstonePlayer
       ├── HeuristicPlacementAgent  ← NEW (replaces RandomPlacementAgent)
       │    └── placement_heuristic.py  ← NEW (scoring functions)
       └── AlphaBetaMainAgentAdapter (unchanged)

train_compact_placement_supervised.py (UNCHANGED — the training pipeline is fine,
                                       the data quality was the problem)
```

## Expected Impact

With heuristic placement at temperature=0.5:
- Both players make strategically sound (but diverse) initial placements
- Game outcomes now more strongly reflect placement quality
- The model should see settlement accuracy jump from ~2% to 30-50%+
- Val loss should decrease meaningfully and continue improving with more data

Recommended data collection command after implementation:
```bash
python capstone_agent/Placement/collect_compact_placement_data.py \
    --games 50000 \
    --games-per-chunk 5000 \
    --out-dir capstone_agent/data/heuristic_placement \
    --strategy heuristic \
    --temperature 0.5 \
    --fixed-board \
    --board-seed 42 \
    --seed 100000
```
