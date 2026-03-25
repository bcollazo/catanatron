"""Observation space utilities and translators between Catanatron and Capstone formats.

This module is the observation-space analogue of ``action_translator.py``.
It provides small helpers for:

- Building the **Catanatron** numeric observation vector from a ``Game``.
- Building the **Capstone** observation vector (your target format) from
  the same ``Game``.
- Treating the Capstone vector as the canonical "our format" representation.

The intent is that any component which only has access to a Catanatron
``Game`` instance can call these helpers to obtain either representation
without depending directly on the Gym env classes.

Public API:
    get_catanatron_observation(game, self_color, map_type="BASE") -> np.ndarray
    get_capstone_observation_vector(game, self_color, opp_color) -> np.ndarray
    catanatron_to_capstone(game, self_color, opp_color, map_type="BASE") -> np.ndarray
"""

from typing import Literal

import numpy as np

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.capstone_features import get_capstone_observation


def get_catanatron_observation(
    game: Game,
    self_color: Color,
    *,
    map_type: str = "BASE",
) -> np.ndarray:
    """Return the original Catanatron numeric observation vector.

    This mirrors the vector representation produced by ``CatanatronEnv``
    when ``representation == "vector"``. It is built by:
      1. Calling ``create_sample(game, self_color)`` to get the full
         feature dict.
      2. Ordering those features using ``get_feature_ordering(...)``.
    """
    num_players = len(game.players)
    features = get_feature_ordering(num_players, map_type)
    sample = create_sample(game, self_color)
    return np.array([float(sample[name]) for name in features], dtype=np.float64)


def get_capstone_observation_vector(
    game: Game,
    self_color: Color,
    opp_color: Color,
) -> np.ndarray:
    """Return the Capstone-style flat observation vector (target format).

    This is a thin wrapper around ``get_capstone_observation`` that
    normalises the result into a NumPy ``float64`` array, matching the
    behaviour of ``CapstoneCatanatronEnv._get_observation``.
    """
    features = get_capstone_observation(game, self_color, opp_color)
    return np.array(features, dtype=np.float64)


def catanatron_to_capstone(
    game: Game,
    self_color: Color,
    opp_color: Color,
    *,
    map_type: str = "BASE",
) -> np.ndarray:
    """Translate from a Catanatron game state to your Capstone observation format.

    Conceptually, this function maps:
        Catanatron board/state  ->  Catanatron features  ->  Capstone features

    In practice, it relies on the shared ``Game`` engine and calls
    :func:`get_capstone_observation_vector`, so it does not require the
    intermediary numeric Catanatron vector to be passed in.

    Args:
        game: The shared Catanatron ``Game`` instance.
        self_color: Color of the learning agent (P0 / "our" perspective).
        opp_color: Color of the opponent (currently assuming 1v1).
        map_type: Logical map type; kept for symmetry with
            :func:`get_catanatron_observation` and future extensions.

    Returns:
        A NumPy ``float64`` array in your Capstone observation-space format.
    """
    # ``map_type`` is not currently needed for Capstone features, but is
    # threaded through for API symmetry and future map-dependent logic.
    _ = map_type
    return get_capstone_observation_vector(game, self_color, opp_color)

