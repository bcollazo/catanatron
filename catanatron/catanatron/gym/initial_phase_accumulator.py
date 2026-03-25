"""Initial placement phase feature logging (before/after each setup action)."""

from catanatron.game import GameAccumulator
from catanatron.models.enums import ActionType
from catanatron.gym.envs.capstone_features import get_capstone_observation


def capstone_obs_for_player(game, self_color):
    other_colors = [c for c in game.state.colors if c != self_color]
    opp_color = other_colors[0] if other_colors else self_color
    return get_capstone_observation(game, self_color, opp_color)


class InitialPhaseFeatureAccumulator(GameAccumulator):
    """
    During the initial road/settlement phase, records capstone feature vectors
    before and after each placement (from the acting player's perspective).

    For 2 players this yields 8 steps (settlement/road alternating); in general
    4 * num_players steps. Stored per finished game as::

        [ [ [obs_before, obs_after], ... ], winner_str ]

    where each obs_* is the list from get_capstone_observation.
    """

    def __init__(self):
        self.initial_phase_by_game = []

    def before(self, game):
        self.current_pairs = []
        self._pending_before = None

    def step(self, game_before_action, action):
        if not game_before_action.state.is_initial_build_phase:
            return
        if action.action_type not in (
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_ROAD,
        ):
            return
        self._pending_before = capstone_obs_for_player(
            game_before_action, action.color
        )

    def step_after(self, game_after_action, action):
        if self._pending_before is None:
            return
        obs_after = capstone_obs_for_player(game_after_action, action.color)
        self.current_pairs.append([self._pending_before, obs_after])
        self._pending_before = None

    def after(self, game):
        winning_color = game.winning_color()
        if winning_color is None:
            return
        self.initial_phase_by_game.append(
            [self.current_pairs, winning_color.value]
        )
