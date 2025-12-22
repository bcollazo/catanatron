"""
Shaped reward function for Catan that provides incremental rewards.

Instead of only rewarding at the end (+1 win, -1 loss), this gives
partial credit for progress during the game.
"""

from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_color,
    get_largest_army,
)


class ShapedRewardFunction:
    """
    Reward function that gives incremental rewards for game progress.

    Rewards:
    - Victory point gain: +1.0 per VP
    - Winning: +10.0 bonus
    - Losing: -10.0 penalty
    - Longest road acquired: +0.5
    - Largest army acquired: +0.5
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset tracked state for a new game."""
        self.prev_vp = 0
        self.prev_has_longest_road = False
        self.prev_has_largest_army = False

    def __call__(self, action, game, p0_color):
        """
        Compute reward for the current step.

        Args:
            action: The action taken
            game: The game object
            p0_color: Player 0's color (BLUE)

        Returns:
            float: The reward for this step
        """
        state = game.state
        reward = 0.0

        # Get current metrics
        current_vp = get_actual_victory_points(state, p0_color)
        longest_road_color = get_longest_road_color(state)
        largest_army_color, _ = get_largest_army(state)

        has_longest_road = longest_road_color == p0_color
        has_largest_army = largest_army_color == p0_color

        # Reward for VP gain
        vp_gain = current_vp - self.prev_vp
        reward += vp_gain * 1.0

        # Reward for acquiring longest road
        if has_longest_road and not self.prev_has_longest_road:
            reward += 0.5
        elif not has_longest_road and self.prev_has_longest_road:
            reward -= 0.5  # Lost longest road

        # Reward for acquiring largest army
        if has_largest_army and not self.prev_has_largest_army:
            reward += 0.5
        elif not has_largest_army and self.prev_has_largest_army:
            reward -= 0.5  # Lost largest army

        # Check for game end
        winning_color = game.winning_color()
        if winning_color is not None:
            if p0_color == winning_color:
                reward += 10.0  # Big bonus for winning
            else:
                reward -= 10.0  # Penalty for losing
            # Reset for next game
            self.reset()
        else:
            # Update tracked state for next step
            self.prev_vp = current_vp
            self.prev_has_longest_road = has_longest_road
            self.prev_has_largest_army = has_largest_army

        return reward


# Create a singleton instance to use
shaped_reward = ShapedRewardFunction()
