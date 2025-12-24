"""
Shared utilities for creating Catanatron environments.
"""

from pathlib import Path

import gymnasium
from sb3_contrib.common.wrappers import ActionMasker

import catanatron.gym
from catanatron import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import simple_reward
from shaped_reward import ShapedRewardFunction


def autodetect_vecnormalize_path(model_path, vecnorm_path=None):
    """
    Resolve VecNormalize stats path from args or a matching file next to the model.

    Returns:
        Tuple of (vecnorm_path or None, auto_detected_bool).
    """
    if vecnorm_path:
        return vecnorm_path, False

    model_path = Path(model_path)
    potential_vecnorm = model_path.parent / f"{model_path.stem}_vecnormalize.pkl"
    if potential_vecnorm.exists():
        return str(potential_vecnorm), True

    return None, False


def make_catan_env(config):
    """
    Factory function to create a Catan environment for vectorization.

    Args:
        config: Dictionary with environment configuration:
            - map_type: Map type for Catan (BASE, MINI, etc.)
            - vps_to_win: Victory points needed to win
            - use_shaped_reward: Whether to use shaped reward function
            - render_mode: Render mode (optional, defaults to "rgb_array")

    Returns:
        Wrapped Catan environment
    """
    reward_fn = (
        ShapedRewardFunction()
        if config.get("use_shaped_reward", True)
        else simple_reward
    )

    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": config.get("map_type", "MINI"),
            "vps_to_win": config.get("vps_to_win", 6),
            "enemies": [ValueFunctionPlayer(Color.RED)],
            "reward_function": reward_fn,
            "render_mode": config.get("render_mode", "rgb_array"),
        },
    )
    env = ActionMasker(env, lambda env: env.unwrapped.action_masks())
    return env
