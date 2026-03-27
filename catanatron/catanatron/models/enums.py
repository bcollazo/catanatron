from enum import IntEnum, auto
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
class NodeRef(IntEnum):
    NORTH = auto()
    NORTHEAST = auto()
    SOUTHEAST = auto()
    SOUTH = auto()
    SOUTHWEST = auto()
    NORTHWEST = auto()


# References an edge from a tile.
class EdgeRef(IntEnum):
    EAST = auto()
    SOUTHEAST = auto()
    SOUTHWEST = auto()
    WEST = auto()
    NORTHWEST = auto()
    NORTHEAST = auto()


class ActionPrompt(IntEnum):
    BUILD_INITIAL_SETTLEMENT = auto()
    BUILD_INITIAL_ROAD = auto()
    PLAY_TURN = auto()
    DISCARD = auto()
    MOVE_ROBBER = auto()
    DECIDE_TRADE = auto()
    DECIDE_ACCEPTEES = auto()


class ActionType(IntEnum):
    """Type of action taken by a player.

    See comments next to each ActionType for the shape of the corresponding
    .value field in Actions of that type.
    """

    ROLL = auto()  # value is None
    MOVE_ROBBER = auto()  # value is (coordinate, Color|None).

    # TODO: None for now to avoid complexity, but should be Resource[].
    DISCARD = auto()  # value is None

    # Building/Buying
    BUILD_ROAD = auto()  # value is edge_id
    BUILD_SETTLEMENT = auto()  # value is node_id
    BUILD_CITY = auto()  # value is node_id
    BUY_DEVELOPMENT_CARD = auto()  # value is None.

    # Dev Card Plays
    PLAY_KNIGHT_CARD = auto()  # value is None
    PLAY_YEAR_OF_PLENTY = auto()  # value is (Resource, Resource)
    PLAY_MONOPOLY = auto()  # value is Resource
    PLAY_ROAD_BUILDING = auto()  # value is None

    # ===== Trade
    # MARITIME_TRADE value is 5-resouce tuple, where last resource is resource asked.
    #   resources in index 2 and 3 might be None, denoting a port-trade.
    MARITIME_TRADE = auto()
    # Domestic Trade (player to player trade)
    # Values for all three is a 10-resource tuple, first 5 is offered freqdeck, last 5 is
    #   receiving freqdeck.
    OFFER_TRADE = auto()
    ACCEPT_TRADE = auto()
    REJECT_TRADE = auto()
    # CONFIRM_TRADE value is 11-tuple. first 10 as in OFFER_TRADE, last is color of accepting player
    CONFIRM_TRADE = auto()
    CANCEL_TRADE = auto()  # value is None

    END_TURN = auto()  # value is None

    def __repr__(self):
        return f"AT.{self.name}"


# TODO: Distinguish between Action and ActionLog?
Action = namedtuple("Action", ["color", "action_type", "value"])
Action.__doc__ = """
Main class to represent action. Should be immutable, and so the 
choice of a namedtuple.

The "value" is a polymorphic field that acts as the "parameters"
for the "action_type". e.g. where to ActionType.BUILD_SETTLEMENT
or who to steal from in a ActionType.MOVE_ROBBER action.

We use this class to represent the _intent_ of say "moving a
robber to Tile (0,0,0) and stealing from Blue".
"""

ActionRecord = namedtuple("ActionRecord", ["action", "result"])
ActionRecord.__doc__ = """
Records an Action along with the result of that action. Useful for
showing an "action log" in a UI, fully replaying a game, or 
undoing actions to a State.

The "result" field is polymorphic depending on the action_type.
- ROLL: result is (int, int) 2 dice rolled
- DISCARD: result is List[Resource] discarded
- MOVE_ROBBER: result is card stolen (Resource|None)
- BUY_DEVELOPMENT_CARD: result is card
- ...for the rest, result is None since they are deterministic actions
"""
