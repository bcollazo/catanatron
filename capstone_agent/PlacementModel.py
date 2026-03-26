"""Lightweight policy network for initial settlement + road placement.

Only two action heads (settlement: 54 nodes, road: 72 edges) plus a
value head for compatibility with the CapstoneAgent interface.  All other
action slots in the 245-dim output are filled with -inf so the mask
zeros them out.

Architecture
------------
Compressor (1259 -> hidden) -> 1 residual block -> dropout -> shared policy stem
    -> settlement head (hidden -> 54)
    -> road head       (hidden -> 72)
    -> value head      (hidden -> 1)
"""

import torch
import torch.nn as nn
from CONSTANTS import (FEATURE_SPACE_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE, ACTION_SPACE_SIZE,
                       ROAD_ACTION_SLICE, SETTLEMENT_ACTION_SLICE)

class PlacementModel(nn.Module):

    VERTEX_ACTION_SIZE = 54  # settlement nodes
    EDGE_ACTION_SIZE = 72    # road edges
    # TODO -> does the placement agent really need the whole feature space? it really only needs a few key features
    def __init__(self, obs_size: int = FEATURE_SPACE_SIZE, hidden_size: int = PLACEMENT_AGENT_HIDDEN_SIZE, dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size

        self.compressor = nn.Linear(obs_size, hidden_size)

        self.residual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x:    (batch, obs_size)
            mask: (batch, 245)

        Returns:
            probs:       (batch, 245)  masked softmax probabilities
            state_value: (batch, 1)
        """
        x = self.compressor(x)

        x = nn.functional.relu(x + self.residual(x))
        x = self.dropout(x)

        state_value = self.value_head(x)

        p = self.policy_stem(x)
        settlement_logits = self.settlement_head(p)
        road_logits = self.road_head(p)

        logits = torch.full(
            (x.size(0), ACTION_SPACE_SIZE), -1e9,
            device=x.device, dtype=x.dtype,
        )
        logits[:, ROAD_ACTION_SLICE] = road_logits
        logits[:, SETTLEMENT_ACTION_SLICE] = settlement_logits

        mask_tensor = torch.as_tensor(mask, device=logits.device)
        logits = logits.masked_fill(mask_tensor == 0, -1e9)
        probs = torch.softmax(logits, dim=-1)

        return probs, state_value
