# reward_functions.py

import numpy as np
from catanatron.state_functions import get_actual_victory_points


def partial_rewards(game, p0_color, vps_to_win):
    """
    Calculate the partial rewards for the game.

    Args:
        game: The game instance.
        p0_color: The color representing the player's position.
        vps_to_win: The victory points required to win the game.

    Returns:
        A float representing the partial reward.
    """
    winning_color = game.winning_color()
    if winning_color is None:
        return 0

    total = 0
    if p0_color == winning_color:
        total += 0.20
    else:
        total -= 0.20
    enemy_vps = [
        get_actual_victory_points(game.state, color)
        for color in game.state.colors
        if color != p0_color
    ]
    enemy_avg_vp = sum(enemy_vps) / len(enemy_vps)
    my_vps = get_actual_victory_points(game.state, p0_color)
    vp_diff = (my_vps - enemy_avg_vp) / (vps_to_win - 1)

    total += 0.80 * vp_diff
    print(f"my_vps = {my_vps} enemy_avg_vp = {enemy_avg_vp} partial_rewards = {total}")
    return total


def mask_fn(env) -> np.ndarray:
    """
    Generates a boolean mask of valid actions for the environment.

    Args:
        env: The environment instance.

    Returns:
        A numpy array of booleans indicating valid actions.
    """
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_actions] = True
    return mask
