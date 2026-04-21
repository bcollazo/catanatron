"""Router-backed Catanatron players for supervised placement data collection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from catanatron.gym.envs.action_translator import (
    capstone_to_action,
    catanatron_action_to_capstone_index,
)
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.models.player import Player
from catanatron.players.minimax import AlphaBetaPlayer

try:
    from ..CapstoneAgent import CapstoneAgent
    from ..CONSTANTS import ACTION_SPACE_SIZE
except ImportError:  # pragma: no cover - supports script-style imports
    from CapstoneAgent import CapstoneAgent
    from CONSTANTS import ACTION_SPACE_SIZE


def get_valid_capstone_actions(playable_actions: Iterable) -> list[int]:
    """Convert engine actions to Capstone action indices."""

    return [catanatron_action_to_capstone_index(action) for action in playable_actions]


def get_capstone_action_mask(playable_actions: Iterable) -> np.ndarray:
    """Build the 245-dim Capstone legality mask from engine playable actions."""

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for idx in get_valid_capstone_actions(playable_actions):
        mask[idx] = 1.0
    return mask


class EnginePlayerMainAgentAdapter:
    """Wrap a native engine `Player` behind the `CapstoneAgent` main-agent API."""

    def __init__(self, engine_player: Player):
        self.engine_player = engine_player
        self.model = None

    def select_action(self, state, mask, *, game=None, playable_actions=None, **_kwargs):
        del state, mask
        if game is None or playable_actions is None:
            raise ValueError(
                "EnginePlayerMainAgentAdapter.select_action requires "
                "`game` and `playable_actions` keyword arguments"
            )

        action = self.engine_player.decide(game, playable_actions)
        capstone_idx = catanatron_action_to_capstone_index(action)
        return capstone_idx, 0.0, 0.0

    def store(self, state, mask, action, log_prob, reward, value, done):
        del state, mask, action, log_prob, reward, value, done

    def train(self, last_value):
        del last_value

    def load(self, path):
        del path

    def save(self, path):
        del path

    def reset_state(self):
        if hasattr(self.engine_player, "reset_state"):
            self.engine_player.reset_state()


class AlphaBetaMainAgentAdapter(EnginePlayerMainAgentAdapter):
    """Main-game adapter that delegates to `AlphaBetaPlayer`."""

    def __init__(self, color, depth: int = 2, prunning: bool = False):
        super().__init__(AlphaBetaPlayer(color, depth=depth, prunning=prunning))


class RouterCapstonePlayer(Player):
    """Engine-facing player that routes through `CapstoneAgent`."""

    def __init__(self, color, placement_agent, main_agent):
        super().__init__(color, is_bot=True)
        self.placement_agent = placement_agent
        self.main_agent = main_agent
        self.router = CapstoneAgent(placement_agent, main_agent)

    def get_action_mask(self, playable_actions: Iterable) -> np.ndarray:
        return get_capstone_action_mask(playable_actions)

    def decide(self, game, playable_actions):
        playable_actions = list(playable_actions)
        opp_color = next(color for color in game.state.colors if color != self.color)
        observation = np.asarray(
            get_capstone_observation(game, self.color, opp_color),
            dtype=np.float32,
        )
        action_mask = self.get_action_mask(playable_actions)
        capstone_action, _, _ = self.router.select_action(
            observation,
            action_mask,
            game=game,
            playable_actions=playable_actions,
        )
        return capstone_to_action(capstone_action, playable_actions)

    def reset_state(self):
        if hasattr(self.placement_agent, "reset_state"):
            self.placement_agent.reset_state()
        if hasattr(self.main_agent, "reset_state"):
            self.main_agent.reset_state()
