from PlacementAgent import PlacementAgent
from MainPlayAgent import MainPlayAgent
import os

from CONSTANTS import (IS_SETTLEMENT_PHASE_FEATURE_INDEX, )

# TODO -> delete
# def _unwrap_env(env):
#     """Walk through gymnasium wrappers to reach the core env."""
#     current = env
#     while hasattr(current, "env"):
#         current = current.env
#     return current


class CapstoneAgent:
    """Routes decisions to a PlacementAgent during the initial build phase
    and to the main MainPlayAgent for the rest of the game.

    Presents the same interface as MainPlayAgent / PlacementAgent so it can
    be used as a drop-in replacement in simulate_game / training_loop.
    """

    def __init__(self, placement_agent: PlacementAgent, main_agent: MainPlayAgent):
        self.placement_agent = placement_agent
        self.main_agent = main_agent
        self._last_was_placement = False

    def _is_placement_phase(self, state) -> bool:
        is_settlement_phase = bool(state[IS_SETTLEMENT_PHASE_FEATURE_INDEX])
        return is_settlement_phase

    def select_action(self, state, mask):
        self._last_was_placement = self._is_placement_phase(state)
        if self._last_was_placement:
            return self.placement_agent.select_action(state, mask)
        return self.main_agent.select_action(state, mask)

    def store(self, state, mask, action, log_prob, reward, value, done):
        if self._last_was_placement:
            self.placement_agent.store(state, mask, action, log_prob, reward, value, done)
        else:
            self.main_agent.store(state, mask, action, log_prob, reward, value, done)

    def train(self, last_value):
        """Run PPO updates on both sub-agents (placement only if it has data)."""
        # TODO -> training happens at a different time for each, as a game only has 4 settlement actions and
        # many more main play actions... this will need to be updated
        self.main_agent.train(last_value)
        self.placement_agent.train(last_value)

    @property
    def model(self):
        """Convenience accessor used by simulate_and_train to compute last_value.

        Returns the main agent's model since last_value is needed for the
        main-game value estimate (placement is already over by end-of-game).
        """
        return self.main_agent.model

    def load(self, main_path, placement_path=None):
        self.main_agent.load(main_path)
        if placement_path is not None:
            self.placement_agent.load(placement_path)

    def save(self, main_path, placement_path=None):
        os.makedirs(os.path.dirname(main_path) or ".", exist_ok=True)
        self.main_agent.save(main_path)
        if placement_path is not None:
            os.makedirs(os.path.dirname(placement_path) or ".", exist_ok=True)
            self.placement_agent.save(placement_path)
