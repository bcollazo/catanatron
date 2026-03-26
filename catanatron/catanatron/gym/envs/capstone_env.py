import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.gym.envs.catanatron_env import (
    to_action_space as to_catanatron_action_space,
)
from catanatron.gym.envs.CapstoneReward import CapstoneReward

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from capstone_agent.CONSTANTS import FEATURE_SPACE_SIZE


BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]

# Our Rendition of the actions array (will need to translate them into actual actions)
# 249 total actions
ACTIONS_ARRAY = [
    # Road, Settlement, City building
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],

    # Robber Movement (stealing automatic in a 2 player game)
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],

    # Card Discard
    # *[(ActionType.DISCARD, resource) for resource in RESOURCES],
    (ActionType.DISCARD, None),
    
    # Bank Trade
    *[(ActionType.MARITIME_TRADE, (resource_give, resource_take)) for resource_give in RESOURCES for resource_take in RESOURCES if resource_give != resource_take],

    # Dev Cards
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (resource1, resource2)) for i, resource1 in enumerate(RESOURCES) for j, resource2 in enumerate(RESOURCES) if j >= i],
    *[(ActionType.PLAY_MONOPOLY, resource) for resource in RESOURCES],

    # Turn Management
    (ActionType.END_TURN, None),
    (ActionType.ROLL, None),
]

ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)
ACTION_TYPES = [i for i in ActionType]

# Imported after ACTIONS_ARRAY is defined to break the circular dependency
# (action_translator imports ACTIONS_ARRAY from this module at load time).
from catanatron.gym.envs.action_translator import (  # noqa: E402
    batch_catanatron_to_capstone,
    capstone_to_action,
)


def to_action_type_space(action_type: ActionType) -> int:
    return ACTION_TYPES.index(action_type)


# NOTE: I think I don't need this if we separate action and action_record nicely...
def normalize_action(action):
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    elif (
        normalized.action_type == ActionType.PLAY_YEAR_OF_PLENTY
        and isinstance(action.value, tuple)
        and len(action.value) == 1
    ):
        # Capstone action space only encodes two-card YOP choices.
        return Action(
            action.color, action.action_type, (action.value[0], action.value[0])
        )
    elif normalized.action_type == ActionType.MARITIME_TRADE:
        # Accept both:
        # - engine playable-actions format: (give, give, give/None, give/None, take)
        # - capstone action-space format: (give, take)
        if len(action.value) == 5:
            give_resource = action.value[0]
            take_resource = action.value[4]
        elif len(action.value) == 2:
            give_resource = action.value[0]
            take_resource = action.value[1]
        else:
            raise ValueError(
                f"Unexpected MARITIME_TRADE value shape: {action.value}"
            )
        return Action(
            action.color, action.action_type, (give_resource, take_resource)
        )
    return normalized


def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint

    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action


class CapstoneCatanatronEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", "full")
        self.reward_manager = CapstoneReward(self.reward_function)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])

        assert all(p.color != Color.BLUE for p in self.enemies)
        self.self_player = Player(Color.BLUE)
        # Tag our controlled policy so GUI action logs can distinguish it from other bots.
        self.self_player.ui_label = "BOT (US)"
        self.opp_color = self.enemies[0].color
        self.players = [self.self_player] + self.enemies
        self.representation = "vector"
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FEATURE_SPACE_SIZE,), dtype=np.float64,
        )

        self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions in capstone action-space indices
        """
        catanatron_indices = list(
            map(to_catanatron_action_space, self.game.playable_actions)
        )
        return batch_catanatron_to_capstone(catanatron_indices)

    def get_action_mask(self) -> np.ndarray:
        """Binary mask of shape (ACTION_SPACE_SIZE,). 1 = valid, 0 = invalid."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        for idx in self.get_valid_actions():
            mask[idx] = 1.0
        return mask

    def step(self, action: int):
        try:
            catan_action = capstone_to_action(
                action, self.game.playable_actions
            )
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(
                valid_actions=self.get_valid_actions(),
                action_mask=self.get_action_mask(),
                is_initial_build_phase=self.game.state.is_initial_build_phase,
            )
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_self_decision()

        observation = self._get_observation()
        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_manager.reward(self.game, self.self_player.color)
        info = dict(
            valid_actions=self.get_valid_actions(),
            action_mask=self.get_action_mask(),
            is_initial_build_phase=self.game.state.is_initial_build_phase,
        )

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()

        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self.invalid_actions_count = 0

        self._advance_until_self_decision()

        self.reward_manager.reset(self.game, self.self_player.color)

        observation = self._get_observation()
        info = dict(
            valid_actions=self.get_valid_actions(),
            action_mask=self.get_action_mask(),
            is_initial_build_phase=self.game.state.is_initial_build_phase,
        )

        return observation, info

    def _get_observation(self) -> np.ndarray:
        features = get_capstone_observation(
            self.game, self.self_player.color, self.opp_color
        )
        return np.array(features, dtype=np.float64)

    def _advance_until_self_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.self_player.color
        ):
            self.game.play_tick()  # will play bot


CapstoneCatanatronEnv.__doc__ = f"""
1v1 environment against a random player

Attributes:
    reward_range: -1 if player lost, 1 if player won, 0 otherwise.
    action_space: Integers from the [0, 244] interval.
        See Action Space table below.
    observation_space: Numeric Feature Vector. See Observation Space table 
        below for quantities. They appear in vector in alphabetical order,
        from the perspective of "current" player (hiding/showing information
        accordingly). self is "current" player. P1 is next in line.
        
        We use the following nomenclature for Tile ids and Node ids.
        Edge ids are self-describing (node-id, node-id) tuples. We also
        use Cube coordinates for tiles (see 
        https://www.redblobgames.com/grids/hexagons/#coordinates)

.. image:: _static/tile-ids.png
  :width: 300
  :alt: Tile Ids
.. image:: _static/node-ids.png
  :width: 300
  :alt: Node Ids

.. list-table:: Action Space
   :widths: 10 100
   :header-rows: 1

   * - Integer
     - Catanatron Action
"""
for i, v in enumerate(ACTIONS_ARRAY):
    CapstoneCatanatronEnv.__doc__ += f"   * - {i}\n     - {v}\n"

CapstoneCatanatronEnv.__doc__ += """

.. list-table:: Observation Space (Raw)
   :widths: 10 50 10 10
   :header-rows: 1

   * - Feature Name
     - Description
     - Number of Features (N=number of players)
     - Type

   * - BANK_<resource>
     - Number of cards of that `resource` in bank
     - 5
     - Integer
   * - BANK_DEV_CARDS
     - Number of development cards in bank
     - 1
     - Integer
    
   * - EDGE<i>_P<j>_ROAD
     - Whether edge `i` is owned by player `j`
     - 72 * N
     - Boolean
   * - NODE<i>_P<j>_SETTLEMENT
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - NODE<i>_P<j>_CITY
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - PORT<i>_IS_<resource>
     - Whether node `i` is port of `resource` (or THREE_TO_ONE).
     - 9 * 6
     - Boolean
   * - TILE<i>_HAS_ROBBER
     - Whether robber is on tile `i`.
     - 19
     - Boolean
   * - TILE<i>_IS_<resource>
     - Whether tile `i` yields `resource` (or DESERT).
     - 19 * 6
     - Boolean
   * - TILE<i>_PROBA
     - Tile `i`'s probability of being rolled.
     - 19
     - Float

   * - IS_DISCARDING
     - Whether current player must discard. For now, there is only 1 
       discarding action (at random), since otherwise action space
       would explode in size.
     - 1
     - Boolean
   * - IS_MOVING_ROBBER
     - Whether current player must move robber (because played knight
       or because rolled a 7).
     - 1
     - Boolean
   * - P<i>_HAS_ROLLED
     - Whether player `i` already rolled dice.
     - N
     - Boolean
   * - P0_HAS_PLAYED _DEVELOPMENT_CARD _IN_TURN
     - Whether current player already played a development card
     - 1
     - Boolean

   * - P0_ACTUAL_VPS
     - Total Victory Points (including Victory Point Development Cards)
     - 1
     - Integer
   * - P0_<resource>_IN_HAND
     - Number of `resource` cards in hand
     - 5
     - Integer
   * - P0_<dev-card>_IN_HAND
     - Number of `dev-card` cards in hand
     - 5
     - Integer
   * - P<i>_NUM_DEVS_IN_HAND
     - Number of hidden development cards player `i` has
     - N
     - Integer
   * - P<i>_NUM_RESOURCES _IN_HAND
     - Number of hidden resource cards player `i` has
     - N
     - Integer

   * - P<i>_HAS_ARMY
     - Whether player <i> has Largest Army
     - N
     - Boolean
   * - P<i>_HAS_ROAD
     - Whether player <i> has Longest Road
     - N
     - Boolean
   * - P<i>_ROADS_LEFT
     - Number of roads pieces player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_SETTLEMENTS_LEFT
     - Number of settlements player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_CITIES_LEFT
     - Number of cities player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_LONGEST_ROAD _LENGTH
     - Length of longest road by player `i`
     - N
     - Integer
   * - P<i>_PUBLIC_VPS
     - Amount of visible victory points for player `i` (i.e.
       doesn't include hidden victory point cards; only army,
       road and settlements/cities).
     - N
     - Integer
   * - P<i>_<dev-card>_PLAYED
     - Amount of `dev-card` cards player `i` has played in game
       (VICTORY_POINT not included).
     - 4 * N
     - Integer
   * - 
     - 
     - 194 * N + 226
     - 
"""
