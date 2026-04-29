# Placement Model & Agent — Input / Output Format

## Overview

The placement subsystem handles the initial settlement + road placement phase
of a 1v1 Catan game (8 total plies: 4 settlements, 4 roads).  It uses a
compact, actor-relative observation that strips away main-game signals the
opening policy doesn't need.

---

## Model Input (`PlacementModel`)

A flat `float32` vector of size **846** (`COMPACT_PLACEMENT_FEATURE_SIZE`),
laid out as two contiguous blocks.

### Node block — 54 nodes x 13 features = 702 floats

For each of the 54 board vertices, in node-id order:

| Feature | Width | Description |
|---------|------:|-------------|
| `self_settlement` | 1 | 1.0 if the acting player occupies this node |
| `opp_settlement` | 1 | 1.0 if the opponent occupies this node |
| `resource_pips` | 5 | Pip count for each resource (wood, brick, sheep, wheat, ore) |
| `is_three_to_one` | 1 | 1.0 if this node has a 3:1 generic port |
| `specific_port` | 5 | One-hot for 2:1 resource-specific port |

### Edge block — 72 edges x 2 features = 144 floats

For each of the 72 board edges:

| Feature | Width | Description |
|---------|------:|-------------|
| `self_road` | 1 | 1.0 if the acting player has a road here |
| `opp_road` | 1 | 1.0 if the opponent has a road here |

The observation is **actor-relative** (self vs. opponent), so the model stays
agnostic to absolute player seat identity.

### Accepted input widths

The `PlacementAgent` auto-detects and accepts either:

- **1259-d** full Capstone observation — projected internally via
  `project_capstone_to_compact_placement()`
- **846-d** compact vector — used directly

---

## Model Output (`PlacementModel.forward`)

Returns a tuple of three tensors:

| Output | Shape | Description |
|--------|-------|-------------|
| `settlement_logits` | `(batch, 54)` | Raw logits over 54 vertex positions |
| `road_logits` | `(batch, 72)` | Raw logits over 72 edge positions |
| `state_value` | `(batch, 1)` | Value estimate in [-1, 1] (Tanh activation) |

Only **one head** is used per decision step — the agent infers which head
from the action mask.

---

## Agent Interface (`PlacementAgent.select_action`)

### Inputs

| Argument | Shape | Description |
|----------|-------|-------------|
| `state` | `(1259,)` or `(846,)` | Full Capstone or compact observation |
| `mask` | `(245,)` | Full Capstone action mask |

### Output

A tuple `(capstone_action, log_prob, value)`:

| Field | Type | Description |
|-------|------|-------------|
| `capstone_action` | `int` | Index in the global 245-d Capstone action space |
| `log_prob` | `float` | Log-probability of the sampled action |
| `value` | `float` | Critic's state-value estimate |

Settlement actions map to Capstone indices **72–125** and road actions to
**0–71**.

### Internal pipeline

1. **Prompt inference** — examines which slice of the 245-d mask has valid
   actions (settlement slice `[72:126]` vs. road slice `[0:72]`).
2. **Observation projection** — projects down to the 846-d compact vector.
3. **Local mask extraction** — pulls the prompt-relevant 54-d or 72-d local
   mask from the global mask.
4. **Forward pass** — runs the model, selects the matching head (settlement
   or road), masks invalid logits with -1e9, samples via `Categorical`.
5. **Action mapping** — converts the local action index back to a global
   Capstone action index.

---

## Architecture

```
Input (846)
  │
  ├─► Encoder: Linear(846→64) → ReLU → LayerNorm → Dropout
  │             Linear(64→64)  → ReLU → LayerNorm
  │
  ├─► Value head:  Linear(64→64) → ReLU → Linear(64→1) → Tanh
  │
  └─► Policy stem: Linear(64→64) → ReLU
        ├─► Settlement head: Linear(64→54)
        └─► Road head:       Linear(64→72)
```

Hidden size defaults to 64 (`PLACEMENT_AGENT_HIDDEN_SIZE`).
