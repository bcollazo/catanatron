from typing import Literal
from functools import lru_cache

from catanatron.models.actions import Action
from catanatron.models.map import get_map_template, NUM_NODES, LandTile
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.board import get_edges
from catanatron.models.player import Color


@lru_cache(maxsize=None)
def get_action_array(map_type: Literal["BASE", "TOURNAMENT", "MINI"]):
    map_template = get_map_template(map_type)
    tile_coordinates = [x for x, y in map_template.topology.items() if y == LandTile]
    actions_array = [
        (ActionType.ROLL, None),
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
        # Move Robber actions include to every tile and from each opponent
        *[
            (ActionType.MOVE_ROBBER, (tile, victim_color))
            for tile in tile_coordinates
            for victim_color in [None] + [color for color in Color]
        ],
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
    return actions_array


ACTION_TYPES = [i for i in ActionType]


def to_action_type_space(action_type: ActionType) -> int:
    return ACTION_TYPES.index(action_type)


def to_action_space(action: Action, map_type: Literal["BASE", "TOURNAMENT", "MINI"]):
    """maps action to space_action equivalent integer"""
    actions_array = get_action_array(map_type)
    return actions_array.index((action.action_type, action.value))


def from_action_space(
    action_int, playable_actions, map_type: Literal["BASE", "TOURNAMENT", "MINI"]
):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches actions_array blueprint
    actions_array = get_action_array(map_type)
    (action_type, value) = actions_array[action_int]
    catan_action = None
    for action in playable_actions:
        if action.action_type == action_type and action.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action
