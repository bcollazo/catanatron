"""Compact policy/value network for opening settlement and road placement."""

import torch
import torch.nn as nn

try:
    from ..CONSTANTS import PLACEMENT_AGENT_HIDDEN_SIZE
    from .placement_features import COMPACT_PLACEMENT_FEATURE_SIZE
except ImportError:  # pragma: no cover - supports script-style imports
    from CONSTANTS import PLACEMENT_AGENT_HIDDEN_SIZE
    from placement_features import COMPACT_PLACEMENT_FEATURE_SIZE


class PlacementModel(nn.Module):
    VERTEX_ACTION_SIZE = 54  # settlement nodes
    EDGE_ACTION_SIZE = 72    # road edges

    def __init__(
        self,
        obs_size: int = COMPACT_PLACEMENT_FEATURE_SIZE,
        hidden_size: int = PLACEMENT_AGENT_HIDDEN_SIZE,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        self.policy_stem = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.settlement_head = nn.Linear(hidden_size, self.VERTEX_ACTION_SIZE)
        self.road_head = nn.Linear(hidden_size, self.EDGE_ACTION_SIZE)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, compact_obs_size)

        Returns:
            settlement_logits: (batch, 54)
            road_logits:       (batch, 72)
            state_value: (batch, 1)
        """
        x = self.encoder(x)

        state_value = self.value_head(x)

        p = self.policy_stem(x)
        settlement_logits = self.settlement_head(p)
        road_logits = self.road_head(p)

        return settlement_logits, road_logits, state_value
