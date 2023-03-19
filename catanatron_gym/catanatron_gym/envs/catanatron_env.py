import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)


BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]
ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    # TODO: One for each tile (and abuse 1v1 setting).
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    (ActionType.END_TURN, None),
]
ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)
ACTION_TYPES = [i for i in ActionType]


def to_action_type_space(action):
    return ACTION_TYPES.index(action.action_type)


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


FEATURES = get_feature_ordering(num_players=2)
NUM_FEATURES = len(FEATURES)

# Highest features is NUM_RESOURCES_IN_HAND which in theory is all resource cards
HIGH = 19 * 5


def simple_reward(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return 0
    else:
        return -1


class CatanatronEnv(gym.Env):
    metadata = {"render.modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        # self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, done, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        done = winning_color is not None
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, done, info

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

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot


CatanatronEnv.__doc__ = f"""
1v1 environment against a random player

Attributes:
    reward_range: -1 if player lost, 1 if player won, 0 otherwise.
    action_space: Integers from the [0, 289] interval. 
        See Action Space table below.
    observation_space: Numeric Feature Vector. See Observation Space table 
        below for quantities. They appear in vector in alphabetical order,
        from the perspective of "current" player (hiding/showing information
        accordingly). P0 is "current" player. P1 is next in line.
        
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
    CatanatronEnv.__doc__ += f"   * - {i}\n     - {v}\n"

CatanatronEnv.__doc__ += """

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
