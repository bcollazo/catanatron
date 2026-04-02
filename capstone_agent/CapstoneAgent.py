from PlacementAgent import PlacementAgent
from MainPlayAgent import MainPlayAgent
import os

import torch

from CONSTANTS import IS_SETTLEMENT_PHASE_FEATURE_INDEX
from device import get_device

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

    def select_action(self, state, mask, **kwargs):
        self._last_was_placement = self._is_placement_phase(state)
        if self._last_was_placement:
            return self.placement_agent.select_action(state, mask, **kwargs)
        return self.main_agent.select_action(state, mask, **kwargs)

    def store(self, state, mask, action, log_prob, reward, value, done, next_obs=None):
        """Store transition. *next_obs* after env.step enables correct placement GAE:
        when setup ends, we mark the transition terminal for the placement buffer so
        advantages do not stitch into the next game's setup."""
        if self._last_was_placement:
            placement_done = done
            if next_obs is not None:
                placement_done = placement_done or not self._is_placement_phase(
                    next_obs
                )
            self.placement_agent.store(
                state, mask, action, log_prob, reward, value, placement_done
            )
        else:
            self.main_agent.store(
                state, mask, action, log_prob, reward, value, done, next_obs=next_obs
            )

    def train(self, last_obs, last_mask):
        """Run PPO on both sub-agents with correct bootstrap values from each critic."""
        device = get_device()
        if last_obs is None or last_mask is None:
            last_v_main = 0.0
            last_v_place = 0.0
        else:
            obs_t = torch.as_tensor(
                last_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask_t = torch.as_tensor(
                last_mask, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                _, v_main = self.main_agent.model(obs_t, mask_t)
                last_v_main = v_main.item()
                place_model = getattr(self.placement_agent, "model", None)
                if place_model is not None:
                    _, v_place = self.placement_agent.model(obs_t, mask_t)
                    last_v_place = v_place.item()
                else:
                    last_v_place = 0.0
        self.main_agent.train(last_v_main)
        self.placement_agent.train(last_v_place)

    @property
    def model(self):
        """Main play policy/value (used by training loops for last-state bootstrap)."""
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
