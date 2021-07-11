import gym
from gym import spaces

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import BaseMap, NUM_NODES, Tile
from catanatron.models.enums import Action, Resource, ActionType
from catanatron.models.board import get_edges
from catanatron_gym.features import create_sample_vector, get_feature_ordering


BASE_TOPOLOGY = BaseMap().topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == Tile]
RESOURCE_LIST = list(Resource)
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
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCE_LIST[j]))
        for i, first_card in enumerate(RESOURCE_LIST)
        for j in range(i, len(RESOURCE_LIST))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCE_LIST],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in Resource],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in Resource
        for j in Resource
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))
        for i in Resource
        for j in Resource
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))
        for i in Resource
        for j in Resource
        if i != j
    ],
    (ActionType.END_TURN, None),
]

ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)


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


class CatanatronEnv(gym.Env):
    metadata = {"render.modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features).
    observation_space = spaces.Box(low=-0, high=HIGH, shape=(NUM_FEATURES,), dtype=int)
    reward_range = (-1, 1)

    def __init__(self):
        pass

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        catan_action = from_action_space(action, self.game.state.playable_actions)
        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = create_sample_vector(self.game, self.p0.color)
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        done = winning_color is not None

        if self.p0.color == winning_color:
            reward = 1
        elif winning_color is None:
            reward = 0
        else:
            reward = -1

        return observation, reward, done, info

    def reset(self):
        p0 = Player(Color.BLUE)
        players = [p0, RandomPlayer(Color.RED)]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0_decision()

        observation = create_sample_vector(self.game, self.p0.color)
        return observation

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_player().color != self.p0.color
        ):
            self.game.play_tick()  # will play bot


CatanatronEnv.__doc__ = f"""
1v1 environment against a random player

Attributes:
    action_space: Space of integers from 0-289 enconding 
        the following table:

.. list-table:: Action Space
   :widths: 25 100
   :header-rows: 1

   * - Integer Representation
     - Catanatron Action
"""
for i, v in enumerate(ACTIONS_ARRAY):
    CatanatronEnv.__doc__ += f"   * - {i}\n     - {v}\n"
