from catanatron import Game
from catanatron.state_functions import get_actual_victory_points
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

    def reset(self, game: Game, self_color):
        self_index = game.state.colors.index(self_color)
        player_state = game.state.player_state

        self.victory_points = get_actual_victory_points(game.state, self_color)
        self.roads_built = NUM_STARTING_ROADS - player_state[f"P{self_index}_ROADS_AVAILABLE"]  # 15 total roads
        self.total_pip_production = sum(_player_production(game, self_color, consider_robber=False))
        self.knights_played =  player_state[f"P{self_index}_PLAYED_KNIGHT"]
        self.dev_cards_bought = _get_dev_cards_bought(player_state, self_index)

    def reward(self, game: Game, self_color):
        return self.reward_func(game, self_color)

    def simple_reward(self, game: Game, self_color):
        winning_color = game.winning_color()
        if self_color == winning_color:
            return 1
        elif winning_color is None:
            return 0
        else:
            return -1
    def full_reward(self, game: Game, self_color):
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

        return reward
