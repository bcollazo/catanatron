"""Modular Graph Neural Network agent for Catan.

Provides CatanGraphEncoder with pluggable convolution type and edge strategy:
  conv_type:      "gcn" | "gat" | "gat_edge" | "nnconv"
  edge_strategy:  "direct" | "preaggregate"

Requirements:
    pip install torch torch-geometric

Architecture:
    Node features (54 x F_node)        Edge features (144 x F_edge, optional)
         |                                    |
    [optional: pre-aggregate edges into nodes]
         |
    GNN layers (message passing over 72-edge board graph)
         |
    Graph-level readout (mean pool over 54 nodes)
         |
    Concatenate with global features (F_global)
         |
    MLP head -> 289 action logits (masked to valid actions)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
except ImportError:
    raise ImportError(
        "This example requires PyTorch Geometric. Install with:\n"
        "  pip install torch torch-geometric"
    )

import gymnasium
import catanatron.gym
from catanatron.gym.graph_env_wrapper import GraphObservationWrapper
from catanatron.gym.graph_features import (
    node_feature_dim,
    augmented_node_feature_dim,
    edge_feature_dim,
    global_feature_dim,
)
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE


CONV_TYPES = ("gcn", "gat", "gat_edge", "nnconv")
EDGE_STRATEGIES = ("direct", "preaggregate")


def _make_conv_layer(conv_type, in_dim, out_dim, edge_dim, num_heads):
    """Factory: instantiate one convolution layer by name."""
    if conv_type == "gcn":
        return GCNConv(in_dim, out_dim)
    elif conv_type == "gat":
        head_dim = out_dim // num_heads
        return GATConv(in_dim, head_dim, heads=num_heads, concat=True)
    elif conv_type == "gat_edge":
        head_dim = out_dim // num_heads
        return GATConv(
            in_dim, head_dim, heads=num_heads, concat=True, edge_dim=edge_dim,
        )
    elif conv_type == "nnconv":
        edge_nn = nn.Sequential(
            nn.Linear(edge_dim, in_dim * out_dim),
        )
        return NNConv(in_dim, out_dim, nn=edge_nn, aggr="mean")
    else:
        raise ValueError(f"Unknown conv_type: {conv_type!r}. Choose from {CONV_TYPES}")


class CatanGraphEncoder(nn.Module):
    """Modular GNN backbone for Catan board state.

    Args:
        conv_type: Convolution variant.
            "gcn"      - GCNConv. Ignores edge features.
            "gat"      - GATConv with multi-head attention. Ignores edge features.
            "gat_edge" - GATConv conditioned on edge features.
            "nnconv"   - NNConv: edge-conditioned message passing via learned MLP.
        edge_strategy: How road info is handled.
            "direct"       - edge_attr passed into conv (only used by gat_edge/nnconv).
            "preaggregate" - road counts already folded into node_features;
                             no edge_attr used during convolution.
        num_players: Number of Catan players (determines feature dims).
        hidden_dim: Width of GNN hidden layers (and MLP).
        num_layers: Number of GNN message-passing layers.
        num_heads: Attention heads (only used by gat / gat_edge).
        dropout: Dropout rate applied after each GNN layer and in the MLP.
    """

    def __init__(
        self,
        conv_type="gcn",
        edge_strategy="direct",
        num_players=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        assert conv_type in CONV_TYPES, f"conv_type must be one of {CONV_TYPES}"
        assert edge_strategy in EDGE_STRATEGIES

        self.conv_type = conv_type
        self.edge_strategy = edge_strategy
        self.num_layers = num_layers
        self.dropout = dropout

        # Determine whether forward() will receive edge_attr
        self._uses_edge_attr = (
            edge_strategy == "direct" and conv_type in ("gat_edge", "nnconv")
        )

        if edge_strategy == "preaggregate":
            in_node = augmented_node_feature_dim(num_players)
        else:
            in_node = node_feature_dim(num_players)

        in_edge = edge_feature_dim(num_players)
        in_global = global_feature_dim(num_players)

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_node if i == 0 else hidden_dim
            self.convs.append(
                _make_conv_layer(conv_type, d_in, hidden_dim, in_edge, num_heads)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # MLP head
        combined_dim = hidden_dim + in_global
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ACTION_SPACE_SIZE),
        )

    def forward(self, node_features, edge_index, global_features,
                edge_attr=None, batch=None):
        """
        Args:
            node_features: (B*54, F_node) — node feature matrix.
            edge_index: (2, B*E) — COO edge indices.
            global_features: (B, F_global) — non-spatial features.
            edge_attr: (B*E, F_edge) or None — edge features for direct strategy.
            batch: (B*54,) — batch assignment for pooling.
        """
        ea = edge_attr if self._uses_edge_attr else None

        x = node_features
        for conv, norm in zip(self.convs, self.norms):
            if ea is not None:
                x = conv(x, edge_index, edge_attr=ea)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = global_mean_pool(x, batch)
        combined = torch.cat([graph_emb, global_features], dim=-1)
        return self.head(combined)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def obs_to_pyg_data(obs: dict, device="cpu") -> Data:
    """Convert one environment observation dict to a PyG Data object."""
    data = Data(
        x=torch.tensor(obs["node_features"], dtype=torch.float32, device=device),
        edge_index=torch.tensor(obs["edge_index"], dtype=torch.long, device=device),
        global_features=torch.tensor(
            obs["global_features"], dtype=torch.float32, device=device
        ).unsqueeze(0),
    )
    if "edge_features" in obs:
        data.edge_attr = torch.tensor(
            obs["edge_features"], dtype=torch.float32, device=device
        )
    return data


def select_action(model, obs, valid_actions, device="cpu"):
    """Forward pass + valid-action masking to sample an action."""
    data = obs_to_pyg_data(obs, device=device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
            data.global_features,
            edge_attr=getattr(data, "edge_attr", None),
            batch=batch,
        )

    mask = torch.full((ACTION_SPACE_SIZE,), float("-inf"), device=device)
    for a in valid_actions:
        mask[a] = 0.0
    masked_logits = logits.squeeze(0) + mask

    probs = F.softmax(masked_logits, dim=-1)
    return torch.multinomial(probs, 1).item()


# ---------------------------------------------------------------------------
# Demo: run one game per configuration
# ---------------------------------------------------------------------------

CONFIGS = [
    {"conv_type": "gcn",      "edge_strategy": "preaggregate", "label": "GCN + preaggregate"},
    {"conv_type": "gat",      "edge_strategy": "preaggregate", "label": "GAT + preaggregate"},
    {"conv_type": "gat_edge", "edge_strategy": "direct",       "label": "GAT+edge + direct"},
    {"conv_type": "nnconv",   "edge_strategy": "direct",       "label": "NNConv + direct"},
]


def play_one_game(conv_type, edge_strategy, hidden_dim=64, num_layers=3):
    """Play a single game with the given GNN configuration."""
    env = GraphObservationWrapper(
        gymnasium.make("catanatron/Catanatron-v0"),
        edge_strategy=edge_strategy,
    )
    model = CatanGraphEncoder(
        conv_type=conv_type,
        edge_strategy=edge_strategy,
        num_players=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.eval()

    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = select_action(model, obs, info["valid_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return steps, total_reward


def main():
    for cfg in CONFIGS:
        label = cfg["label"]
        steps, reward = play_one_game(cfg["conv_type"], cfg["edge_strategy"])
        print(f"[{label:25s}]  steps={steps:4d}  reward={reward}")

    print("\nAll configurations ran successfully.")


if __name__ == "__main__":
    main()
