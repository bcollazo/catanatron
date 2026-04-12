from __future__ import annotations

from typing import Optional

from catanatron import Game
from catanatron.state_functions import get_actual_victory_points
from catanatron.models.enums import Action, ActionType
from catanatron.gym.envs.capstone_features import (
    _player_production,
    _get_dev_cards_bought,
    NUM_STARTING_ROADS,
)

class CapstoneReward:

    WIN_REWARD = 1.0
    LOSE_PENALTY = -1.0
    STEP_PENALTY = -0.001
    GAIN_VP_REWARD = 0.05
    PIP_MULTIPLIER = 1/10
    BUY_DEV_CARD_REWARD = 0.005
    PLAY_KNIGHT_REWARD = 0.005
    BUILD_ROAD_REWARD = 0.001
    # When > 0, only on our MOVE_ROBBER: reward pip production we recover (robber off us)
    # and opponent pips blocked (their sum(..., consider_robber=True) drops).
    ROBBER_SELF_ROBBED_PIP_COEF = 0.01
    ROBBER_OPP_ROBBED_PIP_COEF = 0.01

    """Reward manager for CapstoneCatanatronEnv."""

    def __init__(self, reward_type="simple"):
        REWARD_FUNCS = {
            "simple": self.simple_reward,
            "full": self.full_reward,
        }

        assert reward_type in REWARD_FUNCS, "Reward function must be one of 'simple' or 'full'"
        self.reward_func = REWARD_FUNCS.get(reward_type, self.simple_reward)

        # initialize tracking fields for full_reward (set on reset)
        self.victory_points = 0
        self.roads_built = 0
        self.total_pip_production = 0.0
        self.knights_played = 0
        self.dev_cards_bought = 0
        self.prev_self_robbed_pips = 0.0
        self.prev_opp_robbed_pips = 0.0

    @staticmethod
    def _opponent_color(game: Game, self_color):
        for c in game.state.colors:
            if c != self_color:
                return c
        raise ValueError("expected at least two players for opponent color")

    def reset(self, game: Game, self_color):
        self_index = game.state.colors.index(self_color)
        player_state = game.state.player_state

        self.victory_points = get_actual_victory_points(game.state, self_color)
        self.roads_built = NUM_STARTING_ROADS - player_state[f"P{self_index}_ROADS_AVAILABLE"]  # 15 total roads
        self.total_pip_production = sum(_player_production(game, self_color, consider_robber=False))
        self.knights_played =  player_state[f"P{self_index}_PLAYED_KNIGHT"]
        self.dev_cards_bought = _get_dev_cards_bought(player_state, self_index)
        opp = self._opponent_color(game, self_color)
        self.prev_self_robbed_pips = float(
            sum(_player_production(game, self_color, consider_robber=True))
        )
        self.prev_opp_robbed_pips = float(
            sum(_player_production(game, opp, consider_robber=True))
        )

    def reward(self, game: Game, self_color, acting_action: Optional[Action] = None):
        return self.reward_func(game, self_color, acting_action=acting_action)

    def simple_reward(self, game: Game, self_color, acting_action: Optional[Action] = None):
        winning_color = game.winning_color()
        if self_color == winning_color:
            return 1
        elif winning_color is None:
            return 0
        else:
            return -1
    def full_reward(self, game: Game, self_color, acting_action: Optional[Action] = None):
        reward = self.STEP_PENALTY

        self_index = game.state.colors.index(self_color)
        player_state = game.state.player_state

        # Win/Loss
        winning_color = game.winning_color()
        if self_color == winning_color:
            reward += self.WIN_REWARD
        elif winning_color is not None:
            reward += self.LOSE_PENALTY
        
        # Victory Point Delta
        current_vp = get_actual_victory_points(game.state, self_color)
        vp_delta = current_vp - self.victory_points
        reward += vp_delta * self.GAIN_VP_REWARD  # can also be negative if we lose VPs
        self.victory_points = current_vp

        # building road
        current_roads = NUM_STARTING_ROADS - player_state[f"P{self_index}_ROADS_AVAILABLE"]
        road_delta = current_roads - self.roads_built
        reward += road_delta * self.BUILD_ROAD_REWARD
        self.roads_built = current_roads

        # increasing pips
        current_total_pips = sum(_player_production(game, self_color, consider_robber=False))
        pip_delta = current_total_pips - self.total_pip_production
        reward += pip_delta * self.PIP_MULTIPLIER
        self.total_pip_production = current_total_pips

        # buy dev card
        current_dev = _get_dev_cards_bought(player_state, self_index)
        dev_delta = current_dev - self.dev_cards_bought
        reward += dev_delta * self.BUY_DEV_CARD_REWARD
        self.dev_cards_bought = current_dev

        # play knight
        current_knights = player_state[f"P{self_index}_PLAYED_KNIGHT"]
        knight_delta = current_knights - self.knights_played
        reward += knight_delta * self.PLAY_KNIGHT_REWARD
        self.knights_played = current_knights


        # Robber block/unblock reward
        opp_color = self._opponent_color(game, self_color)
        curr_self_robbed = float(
            sum(_player_production(game, self_color, consider_robber=True))
        )
        curr_opp_robbed = float(
            sum(_player_production(game, opp_color, consider_robber=True))
        )
        if (
            acting_action is not None
            and acting_action.color == self_color
            and acting_action.action_type == ActionType.MOVE_ROBBER
            and (
                self.ROBBER_SELF_ROBBED_PIP_COEF != 0.0
                or self.ROBBER_OPP_ROBBED_PIP_COEF != 0.0
            )
        ):
            d_self = curr_self_robbed - self.prev_self_robbed_pips
            d_opp_block = self.prev_opp_robbed_pips - curr_opp_robbed
            reward += (
                self.ROBBER_SELF_ROBBED_PIP_COEF * d_self
                + self.ROBBER_OPP_ROBBED_PIP_COEF * d_opp_block
            )

        self.prev_self_robbed_pips = curr_self_robbed
        self.prev_opp_robbed_pips = curr_opp_robbed

        return reward
