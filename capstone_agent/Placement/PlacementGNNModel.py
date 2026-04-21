"""Graph Neural Network for opening settlement and road placement."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from catanatron.models.board import get_edges

try:
    from .placement_features import (
        COMPACT_NODE_FEATURE_SIZE,
        STEP_INDICATOR_SIZE,
    )
except ImportError:  # pragma: no cover - supports script-style imports
    from placement_features import (
        COMPACT_NODE_FEATURE_SIZE,
        STEP_INDICATOR_SIZE,
    )


def _build_edge_index():
    """Build directed graph edges for undirected message passing."""

    src, dst = [], []
    for a, b in get_edges():
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def _build_edge_to_node_map():
    """Map each road-edge index to its two endpoint node ids."""

    edge_order = [tuple(sorted(edge)) for edge in get_edges()]
    return torch.tensor(edge_order, dtype=torch.long)


EDGE_INDEX = _build_edge_index()
EDGE_TO_NODES = _build_edge_to_node_map()


class GNNLayer(nn.Module):
    """One attention-style message-passing layer."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        if out_dim % heads != 0:
            raise ValueError(
                f"out_dim={out_dim} must be divisible by attention heads={heads}"
            )

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.randn(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.randn(heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        """Apply one round of batched message passing."""

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        batch_size, num_nodes, _ = x.shape
        heads = self.heads
        head_dim = self.head_dim
        src, dst = edge_index

        projected = self.W(x)
        h = projected.view(batch_size, num_nodes, heads, head_dim)

        h_src = h[:, src]
        h_dst = h[:, dst]

        alpha_src = (h_src * self.attn_src).sum(dim=-1)
        alpha_dst = (h_dst * self.attn_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, negative_slope=0.2)

        alpha = alpha - alpha.amax(dim=1, keepdim=True)
        alpha = alpha.exp()

        alpha_sum = torch.zeros(batch_size, num_nodes, heads, device=x.device)
        alpha_sum.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, heads),
            alpha,
        )
        alpha = alpha / (alpha_sum[:, dst] + 1e-8)
        alpha = self.dropout(alpha)

        msg = h_src * alpha.unsqueeze(-1)

        out = torch.zeros(batch_size, num_nodes, heads, head_dim, device=x.device)
        out.scatter_add_(
            1,
            dst.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, -1, heads, head_dim),
            msg,
        )

        out = out.reshape(batch_size, num_nodes, heads * head_dim)
        out = self.norm(out + projected)
        out = F.relu(out)

        if squeeze_out:
            out = out.squeeze(0)
        return out


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

        self.register_buffer("edge_index", EDGE_INDEX)
        self.register_buffer("edge_to_nodes", EDGE_TO_NODES)

        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.settlement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.road_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, node_features, step_indicators):
        """Return settlement logits, road logits, and state value."""

        batch_size, num_nodes, _ = node_features.shape
        step_expanded = step_indicators.unsqueeze(1).expand(batch_size, num_nodes, -1)
        x = torch.cat([node_features, step_expanded], dim=-1)

        h = self.node_encoder(x)

        for layer in self.gnn_layers:
            h = layer(h, self.edge_index)

        settlement_logits = self.settlement_head(h).squeeze(-1)

        src_nodes = self.edge_to_nodes[:, 0]
        dst_nodes = self.edge_to_nodes[:, 1]
        edge_repr = torch.cat([h[:, src_nodes], h[:, dst_nodes]], dim=-1)
        road_logits = self.road_head(edge_repr).squeeze(-1)

        pooled = h.mean(dim=1)
        state_value = self.value_head(pooled)

        return settlement_logits, road_logits, state_value
