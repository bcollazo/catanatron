from enum import Enum
from collections import namedtuple


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"

    def __repr__(self) -> str:
        return self.value


# Strings are considerably faster than Python Enum's (e.g. at being hashed).
# TODO: Move to ints
WOOD = "WOOD"
BRICK = "BRICK"
SHEEP = "SHEEP"
WHEAT = "WHEAT"
ORE = "ORE"
RESOURCES = [WOOD, BRICK, SHEEP, WHEAT, ORE]

KNIGHT = "KNIGHT"
YEAR_OF_PLENTY = "YEAR_OF_PLENTY"
MONOPOLY = "MONOPOLY"
ROAD_BUILDING = "ROAD_BUILDING"
VICTORY_POINT = "VICTORY_POINT"
DEVELOPMENT_CARDS = [KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT]


class DevelopmentCard(Enum):
    KNIGHT = "KNIGHT"
    YEAR_OF_PLENTY = "YEAR_OF_PLENTY"
    MONOPOLY = "MONOPOLY"
    ROAD_BUILDING = "ROAD_BUILDING"
    VICTORY_POINT = "VICTORY_POINT"


class BuildingType(Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    ROAD = "ROAD"


class ActionPrompt(Enum):
    BUILD_INITIAL_SETTLEMENT = "BUILD_INITIAL_SETTLEMENT"
    BUILD_INITIAL_ROAD = "BUILD_INITIAL_ROAD"
    PLAY_TURN = "PLAY_TURN"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"


class ActionType(Enum):
    """
    Action types are associated with a "value" that can be seen as the "params"
    of such action. They usually hold None for to-be-defined values by the
    execution of the action. After execution, the Actions will be hydrated
    so that they can be used in reproducing a game.
    """

    ROLL = "ROLL"  # value is None. Log instead sets it to (int, int) rolled.
    MOVE_ROBBER = "MOVE_ROBBER"  # value is (coordinate, Color|None). Log has extra element of card stolen.
    DISCARD = "DISCARD"  # value is None|Resource[]. TODO: Should always be Resource[].

    # Building/Buying
    BUILD_ROAD = "BUILD_ROAD"  # value is edge_id
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value is node_id
    BUILD_CITY = "BUILD_CITY"  # value is node_id
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"  # value is None. Log value is card

    # Dev Card Plays
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"  # value is (coordinate, Color|None). Log has extra element of card stolen.
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"  # value is (Resource, Resource)
    PLAY_MONOPOLY = "PLAY_MONOPOLY"  # value is Resource
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"  # value is (edge_id1, edge_id2)

    # Trade
    # MARITIME_TRADE value is 5-resouce tuple, where last resource is resource asked.
    #   resources in index 2 and 3 might be None, denoting a port-trade.
    MARITIME_TRADE = "MARITIME_TRADE"

    # TODO: Domestic trade. Im thinking should contain SUGGEST_TRADE, ACCEPT_TRADE actions...

    END_TURN = "END_TURN"  # value is None


def action_repr(self):
    return f"Action({self.color.value} {self.action_type.value} {self.value})"


# TODO: Distinguish between Action and ActionLog?
Action = namedtuple("Action", ["color", "action_type", "value"])
Action.__repr__ = action_repr
