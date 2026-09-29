"""Gymnasium environment backed by the fair perspective-player boundary."""

from typing import Union

import numpy as np

from catanatron.models.observation_agent import ObservationAgent
from catanatron.models.perspective_player import PerspectivePlayer, build_observation
from catanatron.models.player import Color
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.gym.envs.catanatron_env import CatanatronEnv, MixedObservation


class _GymObservationAgent(ObservationAgent):
    """Placeholder agent used while Gym controls actions directly."""

    def decide_observation(self, observation, playable_actions):
        return playable_actions[0]


class PerspectiveCatanatronEnv(CatanatronEnv):
    """Catanatron Gym environment with observations projected by PerspectivePlayer.

    The public Gymnasium API and vector representations match ``CatanatronEnv``.
    The controlled player is a ``PerspectivePlayer`` and its observation is
    built before converting it to the configured vector or mixed representation.
    """

    def _create_p0(self):
        return PerspectivePlayer(_GymObservationAgent(Color.BLUE))

    def _get_observation(self) -> Union[np.ndarray, MixedObservation]:
        observation = build_observation(self.game, self.p0.color)
        sample = observation.features
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array(
                [sample[i] for i in self.numeric_features], dtype=self.dtype
            )
            return {"board": board_tensor, "numeric": numeric}

        return np.array([sample[i] for i in self.features], dtype=self.dtype)
