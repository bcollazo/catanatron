from enum import Enum
from collections import namedtuple


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


class ActionType(Enum):
    ROLL = "ROLL"  # value is None or rolled value.
    MOVE_ROBBER = "MOVE_ROBBER"  # value is (coordinate, Player|None).
    DISCARD = "DISCARD"  # value is None or discarded cards

    # Building/Buying
    BUILD_ROAD = "BUILD_ROAD"  # value is edge
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value is node
    BUILD_CITY = "BUILD_CITY"
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"

    # Dev Card Plays
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"  # value is (coordinate, player)
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"

    # TRADE: too complicated for now...
    END_TURN = "END_TURN"


Action = namedtuple("Action", ["player", "action_type", "value"])
