from functools import lru_cache
from typing import Tuple, Literal

from catanatron.models.actions import Action
from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.player import Color
from catanatron.models.map import build_map


@lru_cache(maxsize=None)
def get_action_array(
    player_colors: Tuple[Color], map_type: Literal["BASE", "TOURNAMENT", "MINI"]
):
    catan_map = build_map(map_type)
    num_nodes = len(catan_map.land_nodes)
    actions_array = [
        (ActionType.ROLL, None),
        (ActionType.DISCARD, None),
        *[
            (ActionType.BUILD_ROAD, tuple(sorted(edge)))
            for edge in get_edges(catan_map.land_nodes)
        ],
        *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(num_nodes)],
        *[(ActionType.BUILD_CITY, node_id) for node_id in range(num_nodes)],
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
            (ActionType.MOVE_ROBBER, (coordinates, victim_color))
            for coordinates in catan_map.land_tiles.keys()
            for victim_color in [None] + list(player_colors)
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


def to_action_space(
    action: Action,
    player_colors: Tuple[Color],
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
):
    """maps action to space_action equivalent integer"""
    actions_array = get_action_array(player_colors, map_type)
    return actions_array.index((action.action_type, action.value))


def from_action_space(
    action_int,
    color: Color,
    player_colors: Tuple[Color],
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
):
    """maps action_int to catantron.models.actions.Action"""
    actions_array = get_action_array(player_colors, map_type)
    (action_type, value) = actions_array[action_int]
    return Action(color, action_type, value)
