from enum import Enum
from collections import namedtuple


class ActionType(Enum):
    ROLL = "ROLL"

    BUILD_ROAD = "BUILD_ROAD"
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value is (coordinate, noderef)
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
