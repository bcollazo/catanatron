from enum import Enum
from collections import namedtuple
from typing import List, Literal, Final


FastResource = Literal["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
FastDevCard = Literal[
    "KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"
]
FastBuildingType = Literal["SETTLEMENT", "CITY", "ROAD"]

# Strings are considerably faster than Python Enum's (e.g. at being hashed).
# TODO: Move to ints
WOOD: Final = "WOOD"
BRICK: Final = "BRICK"
SHEEP: Final = "SHEEP"
WHEAT: Final = "WHEAT"
ORE: Final = "ORE"
RESOURCES: List[FastResource] = [WOOD, BRICK, SHEEP, WHEAT, ORE]

KNIGHT: Final = "KNIGHT"
YEAR_OF_PLENTY: Final = "YEAR_OF_PLENTY"
MONOPOLY: Final = "MONOPOLY"
ROAD_BUILDING: Final = "ROAD_BUILDING"
VICTORY_POINT: Final = "VICTORY_POINT"
DEVELOPMENT_CARDS: List[FastDevCard] = [
    KNIGHT,
    YEAR_OF_PLENTY,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
]

SETTLEMENT: Final = "SETTLEMENT"
CITY: Final = "CITY"
ROAD: Final = "ROAD"


# Given a tile, the reference to the node.
class NodeRef(Enum):
    NORTH = "NORTH"
    NORTHEAST = "NORTHEAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTH = "SOUTH"
    SOUTHWEST = "SOUTHWEST"
    NORTHWEST = "NORTHWEST"


# References an edge from a tile.
class EdgeRef(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


class ActionPrompt(Enum):
    BUILD_INITIAL_SETTLEMENT = "BUILD_INITIAL_SETTLEMENT"
    BUILD_INITIAL_ROAD = "BUILD_INITIAL_ROAD"
    PLAY_TURN = "PLAY_TURN"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    DECIDE_TRADE = "DECIDE_TRADE"
    DECIDE_ACCEPTEES = "DECIDE_ACCEPTEES"


class ActionType(Enum):
    """Type of action taken by a player.

    See comments next to each ActionType for the shape of the corresponding
    .value field in Actions of that type.
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
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"  # value is None
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"  # value is (Resource, Resource)
    PLAY_MONOPOLY = "PLAY_MONOPOLY"  # value is Resource
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"  # value is None

    # ===== Trade
    # MARITIME_TRADE value is 5-resouce tuple, where last resource is resource asked.
    #   resources in index 2 and 3 might be None, denoting a port-trade.
    MARITIME_TRADE = "MARITIME_TRADE"
    # Domestic Trade (player to player trade)
    # Values for all three is a 10-resource tuple, first 5 is offered freqdeck, last 5 is
    #   receiving freqdeck.
    OFFER_TRADE = "OFFER_TRADE"
    ACCEPT_TRADE = "ACCEPT_TRADE"
    REJECT_TRADE = "REJECT_TRADE"
    # CONFIRM_TRADE value is 11-tuple. first 10 as in OFFER_TRADE, last is color of accepting player
    CONFIRM_TRADE = "CONFIRM_TRADE"
    CANCEL_TRADE = "CANCEL_TRADE"  # value is None

    END_TURN = "END_TURN"  # value is None


def __repr__(self):
    return f"ActionType.{self.value}"


# TODO: Distinguish between Action and ActionLog?
Action = namedtuple("Action", ["color", "action_type", "value"])
Action.__doc__ = """
Main class to represent action. Should be immutable.

The "value" is a polymorphic field that acts as the "parameters"
for the "action_type". e.g. where to ActionType.BUILD_SETTLEMENT
or who to steal from in a ActionType.MOVE_ROBBER action.

We use this class to represent both the _intent_ of say "moving a
robber to Tile (0,0,0) and stealing from Blue" as well as
the final result of such a move. In moves like these where the intent
is not enough to be used to reproduce the game identically,
we use "None"s in the "value" container as placeholders 
for that information needed for fully reproducing a game.
(e.g. card stolen, dev card bought, etc...)

See more on ActionType.
"""
