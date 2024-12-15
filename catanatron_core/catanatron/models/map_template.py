from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple, Type, Union

from catanatron.models.coordinate_system import Direction
from catanatron.models.enums import (
    FastResource,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    EdgeRef,
    NodeRef,
)


EdgeId = Tuple[int, int]
NodeId = int
Coordinate = Tuple[int, int, int]


@dataclass
class LandTile:
    id: int
    resource: Union[FastResource, None]  # None means desert tile
    number: Union[int, None]  # None if desert
    nodes: Dict[NodeRef, NodeId]  # node_ref => node_id
    edges: Dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self):
        return self.id


@dataclass
class Port:
    id: int
    resource: Union[FastResource, None]  # None means desert tile
    direction: Direction
    nodes: Dict[NodeRef, NodeId]  # node_ref => node_id
    edges: Dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self):
        return self.id


@dataclass(frozen=True)
class Water:
    nodes: Dict[NodeRef, int]
    edges: Dict[EdgeRef, EdgeId]


Tile = Union[LandTile, Port, Water]


@dataclass(frozen=True)
class MapTemplate:
    numbers: List[int]
    port_resources: List[Union[FastResource, None]]
    tile_resources: List[Union[FastResource, None]]
    topology: Mapping[
        Coordinate, Union[Type[LandTile], Type[Water], Tuple[Type[Port], Direction]]
    ]


# Small 7-tile map, no ports.
MINI_MAP_TEMPLATE = MapTemplate(
    [3, 4, 5, 6, 8, 9, 10],
    [],
    [WOOD, None, BRICK, SHEEP, WHEAT, WHEAT, ORE],
    {
        # center
        (0, 0, 0): LandTile,
        # first layer
        (1, -1, 0): LandTile,
        (0, -1, 1): LandTile,
        (-1, 0, 1): LandTile,
        (-1, 1, 0): LandTile,
        (0, 1, -1): LandTile,
        (1, 0, -1): LandTile,
        # second layer
        (2, -2, 0): Water,
        (1, -2, 1): Water,
        (0, -2, 2): Water,
        (-1, -1, 2): Water,
        (-2, 0, 2): Water,
        (-2, 1, 1): Water,
        (-2, 2, 0): Water,
        (-1, 2, -1): Water,
        (0, 2, -2): Water,
        (1, 1, -2): Water,
        (2, 0, -2): Water,
        (2, -1, -1): Water,
    },
)

"""Standard 4-player map"""
BASE_MAP_TEMPLATE = MapTemplate(
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
    [
        # These are 2:1 ports
        WOOD,
        BRICK,
        SHEEP,
        WHEAT,
        ORE,
        # These represet 3:1 ports
        None,
        None,
        None,
        None,
    ],
    [
        # Four wood tiles
        WOOD,
        WOOD,
        WOOD,
        WOOD,
        # Three brick tiles
        BRICK,
        BRICK,
        BRICK,
        # Four sheep tiles
        SHEEP,
        SHEEP,
        SHEEP,
        SHEEP,
        # Four wheat tiles
        WHEAT,
        WHEAT,
        WHEAT,
        WHEAT,
        # Three ore tiles
        ORE,
        ORE,
        ORE,
        # One desert
        None,
    ],
    # 3 layers, where last layer is water
    {
        # center
        (0, 0, 0): LandTile,
        # first layer
        (1, -1, 0): LandTile,
        (0, -1, 1): LandTile,
        (-1, 0, 1): LandTile,
        (-1, 1, 0): LandTile,
        (0, 1, -1): LandTile,
        (1, 0, -1): LandTile,
        # second layer
        (2, -2, 0): LandTile,
        (1, -2, 1): LandTile,
        (0, -2, 2): LandTile,
        (-1, -1, 2): LandTile,
        (-2, 0, 2): LandTile,
        (-2, 1, 1): LandTile,
        (-2, 2, 0): LandTile,
        (-1, 2, -1): LandTile,
        (0, 2, -2): LandTile,
        (1, 1, -2): LandTile,
        (2, 0, -2): LandTile,
        (2, -1, -1): LandTile,
        # third (water) layer
        (3, -3, 0): (Port, Direction.WEST),
        (2, -3, 1): Water,
        (1, -3, 2): (Port, Direction.NORTHWEST),
        (0, -3, 3): Water,
        (-1, -2, 3): (Port, Direction.NORTHWEST),
        (-2, -1, 3): Water,
        (-3, 0, 3): (Port, Direction.NORTHEAST),
        (-3, 1, 2): Water,
        (-3, 2, 1): (Port, Direction.EAST),
        (-3, 3, 0): Water,
        (-2, 3, -1): (Port, Direction.EAST),
        (-1, 3, -2): Water,
        (0, 3, -3): (Port, Direction.SOUTHEAST),
        (1, 2, -3): Water,
        (2, 1, -3): (Port, Direction.SOUTHWEST),
        (3, 0, -3): Water,
        (3, -1, -2): (Port, Direction.SOUTHWEST),
        (3, -2, -1): Water,
    },
)
