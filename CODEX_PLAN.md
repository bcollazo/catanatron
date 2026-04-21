# Placement Agent Training Pipeline — Implementation Plan

## Background

This is a Catan AI project. The placement agent handles the initial settlement + road placement phase (4 placements per player in a snake draft). The supervised training pipeline generates games with random placement + AlphaBeta for the rest, keeps the winner's placement actions, and trains a neural network to predict them. **The model currently learns nothing** — validation accuracy matches random baseline (~20%). This plan fixes that.

**Key assumption: Fixed board layout.** All games use the same hex/number/port arrangement. This means AlphaBeta distillation is useless (deterministic = one repeated trajectory), but random placement + winner filtering is viable with enough data and model capacity.

---

## Changes Required

### 1. Increase model capacity

**File:** `capstone_agent/CONSTANTS.py`
- Line 14: Change `PLACEMENT_AGENT_HIDDEN_SIZE = 64` to `PLACEMENT_AGENT_HIDDEN_SIZE = 256`

**File:** `capstone_agent/Placement/PlacementModel.py`
- Line 23: Change default `dropout` from `0.1` to `0.2`

No other changes to PlacementModel.py. The model architecture (encoder → policy stem → settlement/road heads) stays the same, just wider.

### 2. Add step indicator features to the compact observation

The model currently can't distinguish "first settlement placement" from "second settlement placement" except by counting occupied nodes in the observation. Add 4 explicit scalar features: count of self settlements, opponent settlements, self roads, opponent roads already on the board.

**File:** `capstone_agent/Placement/placement_features.py`

Add a new constant:
```python
STEP_INDICATOR_SIZE = 4  # self_settlements, opp_settlements, self_roads, opp_roads
```

Update `COMPACT_PLACEMENT_FEATURE_SIZE`:
```python
COMPACT_PLACEMENT_FEATURE_SIZE = (
    NUM_NODES * COMPACT_NODE_FEATURE_SIZE
    + NUM_EDGES * COMPACT_EDGE_FEATURE_SIZE
    + STEP_INDICATOR_SIZE
)
```

Modify `assemble_compact_placement_observation()` — after assembling the node and edge blocks, append 4 floats:
```python
step_indicators = np.array([
    self_settlements.sum(),   # count of self settlements (0, 1, or 2)
    opp_settlements.sum(),    # count of opp settlements (0, 1, or 2)
    self_roads.sum(),         # count of self roads (0, 1, or 2)
    opp_roads.sum(),          # count of opp roads (0, 1, or 2)
], dtype=np.float32)
```

Concatenate these to the end of the observation vector (after the edge block).

Modify `project_capstone_to_compact_placement()` — same thing, extract the counts from the projected settlement/road arrays and append them.

**File:** `capstone_agent/Placement/placement_supervised_dataset.py`

`reconstruct_compact_x()` calls `assemble_compact_placement_observation()`, which will now automatically include the step indicators since settlements/roads are passed in. No change needed here — it flows through automatically.

### 3. Fix train/val split to be per-game

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

Currently the split (lines 223-227) shuffles individual examples. Examples from the same game share the same board and similar occupancy states, so they leak information across the split.

Change `_load_examples()` to also return a `game_ids` array (one per example). The game_id is already available in each record from `load_chunk_records()`.

In `train()`, replace the per-example split with:
```python
# Get unique game IDs and split by game
unique_game_ids = np.unique(game_ids)
rng = np.random.default_rng(split_seed)
perm = rng.permutation(len(unique_game_ids))
split = int(len(unique_game_ids) * (1 - val_frac))
train_game_ids = set(unique_game_ids[perm[:split]])
val_game_ids = set(unique_game_ids[perm[split:]])

train_idx = np.array([i for i, gid in enumerate(game_ids) if gid in train_game_ids])
val_idx = np.array([i for i, gid in enumerate(game_ids) if gid in val_game_ids])
```

### 4. Add --fixed-board flag to data collection

**File:** `capstone_agent/Placement/collect_compact_placement_data.py`

Currently, `Game(players=[blue, red], seed=game_seed)` randomizes the board per seed (because `State.__init__` calls `CatanMap.from_template(BASE_MAP_TEMPLATE)` which shuffles tiles/numbers/ports using `random.sample` seeded by `random.seed(game_seed)` in `Game.__init__`).

For a fixed board, we need all games to use the same `CatanMap` but different dice sequences.

Add a `--fixed-board` CLI argument (flag, default True) and a `--board-seed` argument (int, default 42).

Implementation:
- At the start of `collect()`, build the shared map once:
  ```python
  if fixed_board:
      import random as _rng
      _rng.seed(board_seed)
      from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
      shared_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
  else:
      shared_map = None
  ```
- Pass `shared_map` through to `_play_one_game()` via the work item tuple
- In `_play_one_game()`, pass `catan_map=shared_map` to `Game(...)`:
  ```python
  game = Game(players=[blue, red], seed=game_seed, catan_map=shared_map)
  ```
  When `catan_map` is provided, `State.__init__` uses it directly instead of generating a new one (this is already supported — see `state.py` line 96: `self.board = Board(catan_map or CatanMap.from_template(BASE_MAP_TEMPLATE))`).

**Important:** The game seed still varies per game. This ensures different dice rolls and different random placement choices, but the board layout is constant.

**Multiprocessing note:** `CatanMap` is read-only after construction, so sharing it across workers via `imap_unordered` is safe. However, since it's passed through `pool.imap_unordered`, it will be pickled. If pickling is an issue, instead pass `board_seed` to each worker and have each worker construct the map identically:
```python
def _play_one_game(args):
    game_seed, ab_depth, board_seed = args
    if board_seed is not None:
        import random as _rng
        _rng.seed(board_seed)
        from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
        fixed_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    else:
        fixed_map = None
    
    random.seed(game_seed)
    np.random.seed(game_seed % (2**32 - 1))
    # ... rest of function ...
    game = Game(players=[blue, red], seed=game_seed, catan_map=fixed_map)
```

Each worker reconstructs the same map from the same `board_seed`, then uses a different `game_seed` for dice/placement randomness.

### 5. Adjust training hyperparameters

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

Change CLI defaults:
- `--lr` default: `1e-3` → `3e-4`
- `--epochs` default: `30` → `50`
- `--batch-size` default: `256` → `512`
- `--hidden-size` default: `64` → `256` (to match the CONSTANTS change)

Add early stopping with patience. After the epoch loop computes `val_metrics`, track epochs since improvement:
```python
patience = 10
epochs_without_improvement = 0

# Inside epoch loop, after computing val_metrics:
if val_metrics["loss"] < best_val_loss:
    best_val_loss = val_metrics["loss"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break
```

Add `--patience` CLI argument (int, default 10).

### 6. Create PlacementGNNModel

**New file:** `capstone_agent/Placement/PlacementGNNModel.py`

This is a Graph Neural Network that operates on the Catan board graph. Instead of flattening the 54-node and 72-edge features into a single vector, it processes them as structured graph data.

```python
"""Graph Neural Network for opening settlement and road placement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from catanatron.models.board import STATIC_GRAPH, get_edges

try:
    from .placement_features import (
        NUM_NODES, NUM_EDGES,
        COMPACT_NODE_FEATURE_SIZE, COMPACT_EDGE_FEATURE_SIZE,
        STEP_INDICATOR_SIZE,
    )
except ImportError:
    from placement_features import (
        NUM_NODES, NUM_EDGES,
        COMPACT_NODE_FEATURE_SIZE, COMPACT_EDGE_FEATURE_SIZE,
        STEP_INDICATOR_SIZE,
    )


def _build_edge_index():
    """Build a (2, num_edges) tensor of directed edges from STATIC_GRAPH.
    
    Includes both directions (i→j and j→i) for undirected message passing.
    Only includes edges between land nodes (0-53).
    """
    edges = get_edges()  # list of (node_a, node_b) tuples, land nodes only
    src, dst = [], []
    for a, b in edges:
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def _build_edge_to_node_map():
    """Map each of the 72 road-edge indices to their (src, dst) node pair.
    
    Returns a (72, 2) tensor where row i contains the two node IDs of edge i.
    """
    edge_order = [tuple(sorted(edge)) for edge in get_edges()]
    return torch.tensor(edge_order, dtype=torch.long)


# Precompute once at module load
EDGE_INDEX = _build_edge_index()        # (2, num_directed_edges)
EDGE_TO_NODES = _build_edge_to_node_map()  # (72, 2)


class GNNLayer(nn.Module):
    """One layer of message passing with attention (simplified GAT)."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.randn(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.randn(heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: (num_nodes, in_dim)
            edge_index: (2, num_edges) — directed edges
        Returns:
            (num_nodes, out_dim)
        """
        N = x.size(0)
        H, D = self.heads, self.head_dim

        h = self.W(x).view(N, H, D)  # (N, H, D)

        src, dst = edge_index  # each (num_edges,)

        # Attention scores
        alpha_src = (h[src] * self.attn_src).sum(dim=-1)  # (E, H)
        alpha_dst = (h[dst] * self.attn_dst).sum(dim=-1)  # (E, H)
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)

        # Softmax per destination node
        alpha = alpha - alpha.max()
        alpha = alpha.exp()
        alpha_sum = torch.zeros(N, H, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[dst] + 1e-8)
        alpha = self.dropout(alpha)

        # Aggregate messages
        msg = h[src] * alpha.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        out = out.reshape(N, H * D)
        out = self.norm(out + self.W(x))  # residual
        return F.relu(out)


class PlacementGNNModel(nn.Module):
    """GNN-based placement policy/value network."""

    VERTEX_ACTION_SIZE = 54
    EDGE_ACTION_SIZE = 72

    def __init__(
        self,
        node_in_dim=COMPACT_NODE_FEATURE_SIZE + STEP_INDICATOR_SIZE,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        # Register precomputed graph structure as buffers (move to device with model)
        self.register_buffer("edge_index", EDGE_INDEX)
        self.register_buffer("edge_to_nodes", EDGE_TO_NODES)

        # Node feature projection (13 node features + 4 step indicators → hidden)
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Settlement head: per-node score
        self.settlement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Road head: per-edge score from endpoint node embeddings
        self.road_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value head: global graph readout
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, node_features, step_indicators):
        """
        Args:
            node_features: (batch, 54, 13) per-node features
            step_indicators: (batch, 4) step indicator features

        Returns:
            settlement_logits: (batch, 54)
            road_logits: (batch, 72)
            state_value: (batch, 1)
        """
        B, N, _ = node_features.shape

        # Broadcast step indicators to each node: (B, 54, 13+4)
        step_expanded = step_indicators.unsqueeze(1).expand(B, N, -1)
        x = torch.cat([node_features, step_expanded], dim=-1)

        # Process each graph in the batch
        settlement_logits_list = []
        road_logits_list = []
        values_list = []

        for b in range(B):
            h = self.node_encoder(x[b])  # (54, hidden)

            for layer in self.gnn_layers:
                h = layer(h, self.edge_index)  # (54, hidden)

            # Settlement logits: per-node
            s_logits = self.settlement_head(h).squeeze(-1)  # (54,)

            # Road logits: concatenate endpoint embeddings
            src_nodes = self.edge_to_nodes[:, 0]  # (72,)
            dst_nodes = self.edge_to_nodes[:, 1]  # (72,)
            edge_repr = torch.cat([h[src_nodes], h[dst_nodes]], dim=-1)  # (72, 2*hidden)
            r_logits = self.road_head(edge_repr).squeeze(-1)  # (72,)

            # Value: mean-pool node embeddings
            v = self.value_head(h.mean(dim=0, keepdim=True))  # (1, 1)

            settlement_logits_list.append(s_logits)
            road_logits_list.append(r_logits)
            values_list.append(v.squeeze(0))

        return (
            torch.stack(settlement_logits_list),  # (B, 54)
            torch.stack(road_logits_list),         # (B, 72)
            torch.stack(values_list),              # (B, 1)
        )
```

**Important architectural notes:**
- The GNN model does NOT take the flattened 846-d (now 850-d with step indicators) observation. It takes structured inputs: `(batch, 54, 13)` node features and `(batch, 4)` step indicators.
- Edge features (self_road, opp_road) from the compact observation are NOT used as GNN edge attributes in this design. Instead, the road occupancy is visible through the node-level occupancy features. If needed, edge features can be added as edge attributes in a future iteration.
- The `forward()` loops over the batch dimension. For larger batches, this can be optimized using `torch_geometric`'s batched graph representation, but for placement training (small batches, fast forward pass), the loop is fine.

### 7. Update data pipeline for GNN model support

**File:** `capstone_agent/Placement/placement_features.py`

Add a function to return structured graph inputs instead of the flat vector:
```python
def assemble_graph_placement_inputs(
    self_settlements, opp_settlements, static_node_features,
    self_roads, opp_roads,
):
    """Return structured graph inputs for the GNN model.
    
    Returns:
        node_features: (54, 13) — same per-node features as compact obs
        step_indicators: (4,) — counts of settlements and roads
    """
    self_settlements = np.asarray(self_settlements, dtype=np.float32).reshape(NUM_NODES)
    opp_settlements = np.asarray(opp_settlements, dtype=np.float32).reshape(NUM_NODES)
    static_node_features = np.asarray(static_node_features, dtype=np.float32).reshape(
        NUM_NODES, STATIC_NODE_FEATURE_SIZE
    )
    self_roads = np.asarray(self_roads, dtype=np.float32).reshape(NUM_EDGES)
    opp_roads = np.asarray(opp_roads, dtype=np.float32).reshape(NUM_EDGES)

    node_features = np.concatenate([
        self_settlements[:, None],
        opp_settlements[:, None],
        static_node_features,
    ], axis=1)  # (54, 13)
    
    step_indicators = np.array([
        self_settlements.sum(),
        opp_settlements.sum(),
        self_roads.sum(),
        opp_roads.sum(),
    ], dtype=np.float32)  # (4,)
    
    return node_features, step_indicators
```

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

Add a `--model-type` CLI argument with choices `mlp` (default) and `gnn`.

When `model_type == "gnn"`:
- In `_load_examples()`: store the raw observation arrays as before (the flat vector), but ALSO reshape them into node features and step indicators at training time
- Instantiate `PlacementGNNModel` instead of `PlacementModel`
- In `_log_probs_for_batch()` and `_predict_local_actions()`: reshape the flat obs into `(batch, 54, 13)` node features and `(batch, 4)` step indicators before passing to the model
- The masking and loss computation remain identical — the GNN outputs the same shape logits as the MLP

Helper to reshape flat obs for GNN:
```python
def _flat_obs_to_graph(obs_t):
    """Reshape flat (B, 850) observation to GNN inputs.
    
    Returns:
        node_features: (B, 54, 13)
        step_indicators: (B, 4)
    """
    B = obs_t.shape[0]
    node_block = obs_t[:, :NUM_NODES * COMPACT_NODE_FEATURE_SIZE]
    node_features = node_block.reshape(B, NUM_NODES, COMPACT_NODE_FEATURE_SIZE)
    step_indicators = obs_t[:, -STEP_INDICATOR_SIZE:]
    return node_features, step_indicators
```

**File:** `capstone_agent/Placement/PlacementAgent.py`

Add `model_type` parameter to the `PlacementAgent.__init__()`:
```python
def __init__(self, obs_size=FEATURE_SPACE_SIZE, hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE, model_type="mlp"):
```

When `model_type == "gnn"`:
- Import and instantiate `PlacementGNNModel` instead of `PlacementModel`
- In the forward pass methods (`select_action`, `_evaluate_actions`), reshape the compact observation into node features + step indicators before calling the model
- The rest of the logic (masking, action selection, loss computation) stays the same

---

## Testing

After implementing all changes, verify:

1. **Import check**: `python -c "from capstone_agent.Placement.PlacementModel import PlacementModel; print(PlacementModel().encoder[0].in_features)"` should print the new `COMPACT_PLACEMENT_FEATURE_SIZE` (850 = 846 + 4).

2. **Feature size consistency**: Run a quick smoke test that generates one game, reconstructs examples, and checks that `example.x.shape[0] == COMPACT_PLACEMENT_FEATURE_SIZE`.

3. **GNN forward pass**: Create a `PlacementGNNModel`, pass random `(2, 54, 13)` node features and `(2, 4)` step indicators, verify output shapes are `(2, 54)`, `(2, 72)`, `(2, 1)`.

4. **Per-game split**: Verify that no game_id appears in both train and val sets.

5. **Fixed board**: Run two games with different seeds but `--fixed-board` and verify `static_node_features` are identical.

---

## File Summary

| File | Action |
|------|--------|
| `capstone_agent/CONSTANTS.py` | Change `PLACEMENT_AGENT_HIDDEN_SIZE` 64 → 256 |
| `capstone_agent/Placement/PlacementModel.py` | Change default dropout 0.1 → 0.2 |
| `capstone_agent/Placement/placement_features.py` | Add `STEP_INDICATOR_SIZE`, update `COMPACT_PLACEMENT_FEATURE_SIZE`, add step indicators to `assemble_compact_placement_observation()` and `project_capstone_to_compact_placement()`, add `assemble_graph_placement_inputs()` |
| `capstone_agent/Placement/train_compact_placement_supervised.py` | Per-game split, new defaults (lr=3e-4, epochs=50, batch=512, hidden=256), early stopping with patience, `--model-type` argument, GNN support |
| `capstone_agent/Placement/collect_compact_placement_data.py` | Add `--fixed-board` flag and `--board-seed`, pass shared CatanMap to all games |
| `capstone_agent/Placement/PlacementGNNModel.py` | **NEW** — GNN model with GAT layers, per-node settlement head, per-edge road head |
| `capstone_agent/Placement/PlacementAgent.py` | Add `model_type` parameter, support GNN model in inference path |
