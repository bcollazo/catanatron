# Codex Plan: MPS (Apple Silicon) Training Optimization

## Context

The training script `capstone_agent/Placement/train_compact_placement_supervised.py` and the GNN model `capstone_agent/Placement/PlacementGNNModel.py` already detect and use the MPS backend on Apple Silicon. However, several patterns prevent full GPU utilization on an M4 Max chip. This plan addresses the four highest-impact issues in priority order.

---

## Change 1 (CRITICAL): Batch the GNN forward pass — eliminate the Python loop

**File:** `capstone_agent/Placement/PlacementGNNModel.py`

**Problem:** The `forward()` method (line 146) loops `for batch_idx in range(batch_size)`, processing one graph at a time. With batch_size=512, this means 512 sequential Python-level forward passes. This is the single biggest performance bottleneck.

**Fix:** Since every game uses the same graph topology (same `edge_index`), the entire batch can be processed in parallel by treating the batch as an extra leading dimension. All the GAT operations can be vectorized over the batch.

Replace the current `forward()` method with a fully batched version:

```python
def forward(self, node_features, step_indicators):
    """Return settlement logits, road logits, and state value.
    
    Args:
        node_features: (batch, 54, node_feature_dim)
        step_indicators: (batch, 4)
    Returns:
        settlement_logits: (batch, 54)
        road_logits: (batch, 72)
        state_value: (batch, 1)
    """
    batch_size, num_nodes, _ = node_features.shape
    step_expanded = step_indicators.unsqueeze(1).expand(batch_size, num_nodes, -1)
    x = torch.cat([node_features, step_expanded], dim=-1)

    # Encode: (B, 54, node_in_dim) -> (B, 54, hidden)
    h = self.node_encoder(x)

    # GNN layers: all operate on (B, 54, hidden)
    for layer in self.gnn_layers:
        h = layer(h, self.edge_index)

    # Settlement head: per-node scoring -> (B, 54)
    settlement_logits = self.settlement_head(h).squeeze(-1)

    # Road head: concatenate endpoint embeddings -> (B, 72)
    src_nodes = self.edge_to_nodes[:, 0]  # (72,)
    dst_nodes = self.edge_to_nodes[:, 1]  # (72,)
    edge_repr = torch.cat([h[:, src_nodes], h[:, dst_nodes]], dim=-1)  # (B, 72, hidden*2)
    road_logits = self.road_head(edge_repr).squeeze(-1)  # (B, 72)

    # Value head: mean-pool -> (B, 1)
    pooled = h.mean(dim=1)  # (B, hidden)
    state_value = self.value_head(pooled)  # (B, 1)

    return settlement_logits, road_logits, state_value
```

This also requires **batching the GNNLayer**. The current `GNNLayer.forward()` expects `x` of shape `(num_nodes, hidden)` — it needs to handle `(batch, num_nodes, hidden)`.

Replace `GNNLayer.forward()` with:

```python
def forward(self, x, edge_index):
    """Apply one round of message passing.
    
    Args:
        x: (batch, num_nodes, in_dim) or (num_nodes, in_dim)
        edge_index: (2, num_edges)
    Returns:
        out: same shape as x
    """
    # Handle both batched and unbatched input
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B, num_nodes, _ = x.shape
    heads = self.heads
    head_dim = self.head_dim
    src, dst = edge_index  # (E,), (E,)

    projected = self.W(x)  # (B, N, out_dim)
    h = projected.view(B, num_nodes, heads, head_dim)  # (B, N, H, D)

    # Gather source and destination node features: (B, E, H, D)
    h_src = h[:, src]
    h_dst = h[:, dst]

    # Attention scores: (B, E, H)
    alpha_src = (h_src * self.attn_src).sum(dim=-1)
    alpha_dst = (h_dst * self.attn_dst).sum(dim=-1)
    alpha = F.leaky_relu(alpha_src + alpha_dst, negative_slope=0.2)

    # Softmax per destination node
    alpha = alpha - alpha.amax(dim=1, keepdim=True)  # stability
    alpha = alpha.exp()

    # Sum of attention weights per destination node: (B, N, H)
    alpha_sum = torch.zeros(B, num_nodes, heads, device=x.device)
    alpha_sum.scatter_add_(
        1,
        dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, heads),
        alpha,
    )
    alpha = alpha / (alpha_sum[:, dst] + 1e-8)
    alpha = self.dropout(alpha)

    # Weighted messages: (B, E, H, D)
    msg = h_src * alpha.unsqueeze(-1)

    # Aggregate messages at destination nodes: (B, N, H, D)
    out = torch.zeros(B, num_nodes, heads, head_dim, device=x.device)
    out.scatter_add_(
        1,
        dst.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, heads, head_dim),
        msg,
    )

    out = out.reshape(B, num_nodes, heads * head_dim)
    out = self.norm(out + projected)
    out = F.relu(out)

    if squeeze_out:
        out = out.squeeze(0)
    return out
```

**Key insight:** `scatter_add_` works fine with a batch dimension as long as the scatter is over dimension 1 (the node dimension). The `edge_index` is the same for every batch element, so we just broadcast it.

**Verification:** After this change, run:
```python
python3 -c "
import torch
from capstone_agent.Placement.PlacementGNNModel import PlacementGNNModel

model = PlacementGNNModel(hidden_dim=128)
node_feat = torch.randn(4, 54, 13)
step_ind = torch.randn(4, 4)
s, r, v = model(node_feat, step_ind)
assert s.shape == (4, 54), f'settlement shape: {s.shape}'
assert r.shape == (4, 72), f'road shape: {r.shape}'
assert v.shape == (4, 1), f'value shape: {v.shape}'
print('Batched GNN forward pass OK')
"
```

---

## Change 2: Add `torch.compile()` for the model

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

After the model is created and moved to device (around line 296-301), and after weights are loaded if resuming (around line 305), add:

```python
# Compile the model for MPS/CUDA kernel fusion (PyTorch 2.x)
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
        print(f"Model compiled with torch.compile()", flush=True)
    except Exception as e:
        print(f"torch.compile() not available, continuing without: {e}", flush=True)
```

**Important:** Place this AFTER the `model.load_state_dict(...)` call for resume, but BEFORE the optimizer is created (because the optimizer references `model.parameters()`).

Add a CLI flag to opt out:

```python
parser.add_argument(
    "--no-compile",
    action="store_true",
    help="Disable torch.compile() optimization",
)
```

And wrap the compile in `if not args.no_compile:`.

Pass `no_compile` through to the `train()` function as a parameter.

**Note:** `torch.compile()` on MPS may not support all ops (particularly scatter_add_ in the GNN). If it fails at runtime for GNN models, catch the error gracefully and fall back to eager mode. The MLP model should always work with compile.

---

## Change 3: Mixed precision with `torch.autocast`

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

Wrap the forward pass and loss computation in `torch.autocast`. In the training loop (around lines 434-442):

```python
with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
    log_probs, _ = _log_probs_for_batch(
        model, b_obs, b_prompt, b_mask, b_act, model_type,
    )
    loss = (-log_probs * b_weight).mean()
```

Similarly wrap the eval forward passes in `_evaluate()` and the train_acc computation.

**Important notes for MPS:**
- Do NOT use `torch.cuda.amp.GradScaler` — it's CUDA-only. On MPS, autocast alone is sufficient without a scaler. The pattern is:
  ```python
  use_amp = device.type in ("cuda", "mps")
  # In training loop:
  with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      ...
  # No GradScaler needed for MPS
  loss.backward()  # outside autocast
  ```
- `optimizer.step()` and `loss.backward()` should remain OUTSIDE the autocast context
- `scatter_add_` in the GNN may not support float16 on MPS. If this causes errors, set `enabled=(use_amp and model_type != "gnn")` as a fallback. Test both MLP and GNN paths.

Add a CLI flag:

```python
parser.add_argument(
    "--no-amp",
    action="store_true",
    help="Disable automatic mixed precision",
)
```

---

## Change 4: Move batch permutation indexing to GPU

**File:** `capstone_agent/Placement/train_compact_placement_supervised.py`

Currently (line 420):
```python
batch_perm = torch.as_tensor(rng.permutation(len(train_idx)), dtype=torch.long)
```

This creates the permutation on CPU. Then `train_idx[batch_perm[start:start+batch_size]]` indexes into the numpy array on CPU, and the result indexes into GPU tensors, triggering a CPU→GPU sync per batch.

Fix: pre-convert `train_idx` to a GPU tensor and do all indexing on GPU:

At line ~318-325, after creating `train_idx` and `val_idx` as numpy arrays, convert them:

```python
train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
```

Then in the epoch loop:
```python
batch_perm = torch.as_tensor(
    rng.permutation(len(train_idx_t)), dtype=torch.long, device=device
)
```

And the batch slicing becomes:
```python
batch_rows = train_idx_t[batch_perm[start : start + batch_size]]
```

Also update the val eval to use `val_idx_t`:
```python
vi = val_idx_t  # already on device
```

This eliminates the CPU→GPU sync on every batch.

---

## What NOT to change

- **Don't use DataLoader with num_workers>0** — the entire dataset fits in memory already and is pre-loaded to GPU. A DataLoader would add overhead copying data back from CPU workers. The current direct-indexing approach is correct for this dataset size.
- **Don't move data to CPU** — keeping everything in MPS unified memory is the right call on Apple Silicon since CPU and GPU share the same physical memory.
- **Don't change float32 storage** — the observation tensors should remain float32. Autocast handles the precision reduction only during forward passes.

---

## Verification

1. `python -m py_compile capstone_agent/Placement/PlacementGNNModel.py` — passes
2. `python -m py_compile capstone_agent/Placement/train_compact_placement_supervised.py` — passes
3. Batched GNN test (see Change 1 verification command above) — passes
4. Quick MLP smoke test to confirm torch.compile + autocast work:
   ```bash
   python capstone_agent/Placement/train_compact_placement_supervised.py \
       --data capstone_agent/data/compact_placement_smoke \
       --out /tmp/mps_test_mlp.pt \
       --epochs 3 \
       --batch-size 64 \
       --metrics-log /tmp/mps_test_mlp_metrics.jsonl
   ```
   Should complete without errors and print "Model compiled with torch.compile()".
5. Quick GNN smoke test:
   ```bash
   python capstone_agent/Placement/train_compact_placement_supervised.py \
       --data capstone_agent/data/compact_placement_smoke \
       --out /tmp/mps_test_gnn.pt \
       --model-type gnn \
       --epochs 3 \
       --batch-size 64 \
       --metrics-log /tmp/mps_test_gnn_metrics.jsonl
   ```
   Should complete without errors. If torch.compile or autocast fail for GNN, they should fall back gracefully with a printed warning (not crash).

## Expected impact

| Change | Expected speedup | Risk |
|--------|-----------------|------|
| Batched GNN | 10-50x for GNN training | Medium — scatter_add_ batching needs careful index expansion |
| torch.compile | 1.3-2x for MLP, uncertain for GNN | Low — wrapped in try/except |
| Mixed precision | 1.5-2x for MLP | Low — MPS autocast is mature in PyTorch 2.10 |
| GPU indexing | 1.1-1.2x (eliminates sync stalls) | Very low — straightforward tensor move |
