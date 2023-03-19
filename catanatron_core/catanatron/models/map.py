import typing
from dataclasses import dataclass
import random
from enum import Enum
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, List, Literal, Mapping, Set, Tuple, Type, Union

from catanatron.models.coordinate_system import Direction, add, UNIT_VECTORS
from catanatron.models.enums import (
    FastResource,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
)

NUM_NODES = 54
NUM_EDGES = 72
NUM_TILES = 19


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


EdgeId = Tuple[int, int]
NodeId = int
Coordinate = Tuple[int, int, int]


class LandTile:
    def __init__(
        self,
        tile_id: int,
        resource: Union[FastResource, None],
        number: Union[int, None],
        nodes: Dict[NodeRef, NodeId],
        edges: Dict[EdgeRef, EdgeId],
    ):
        self.id = tile_id

        self.resource = resource  # None means desert tile
        self.number = number  # None if desert

        self.nodes = nodes  # node_ref => node_id
        self.edges = edges  # edge_ref => edge

    def __repr__(self):
        if self.resource is None:
            return "Tile:Desert"
        return f"Tile:{self.number}{self.resource}"


class Port:
    def __init__(
        self, port_id, resource: Union[FastResource, None], direction, nodes, edges
    ):
        self.id = port_id
        self.resource = resource  # None means its a 3:1 port.
        self.direction = direction
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Port:" + str(self.resource)


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


class CatanMap:
    """Represents a randomly initialized map."""

    def __init__(
        self,
        tiles: Dict[Coordinate, Tile] = dict(),
        land_tiles: Dict[Coordinate, LandTile] = dict(),
        port_nodes: Dict[Union[FastResource, None], Set[int]] = dict(),
        land_nodes: FrozenSet[NodeId] = frozenset(),
        adjacent_tiles: Dict[int, List[LandTile]] = dict(),
        node_production: Dict[NodeId, Counter] = dict(),
        tiles_by_id: Dict[int, LandTile] = dict(),
        ports_by_id: Dict[int, Port] = dict(),
    ):
        self.tiles = tiles
        self.land_tiles = land_tiles
        self.port_nodes = port_nodes
        self.land_nodes = land_nodes
        self.adjacent_tiles = adjacent_tiles
        self.node_production = node_production
        self.tiles_by_id = tiles_by_id
        self.ports_by_id = ports_by_id

    @staticmethod
    def from_template(map_template: MapTemplate):
        tiles = initialize_tiles(map_template)

        return CatanMap.from_tiles(tiles)

    @staticmethod
    def from_tiles(tiles: Dict[Coordinate, Tile]):
        self = CatanMap()
        self.tiles = tiles

        self.land_tiles = {
            k: v for k, v in self.tiles.items() if isinstance(v, LandTile)
        }

        # initialize auxiliary data structures for fast-lookups
        self.port_nodes = init_port_nodes_cache(self.tiles)

        land_nodes_list = map(lambda t: set(t.nodes.values()), self.land_tiles.values())
        self.land_nodes = frozenset(set.union(*land_nodes_list))

        # TODO: Rename to self.node_to_tiles
        self.adjacent_tiles = init_adjacent_tiles(self.land_tiles)
        self.node_production = init_node_production(self.adjacent_tiles)
        self.tiles_by_id = {
            t.id: t for t in self.tiles.values() if isinstance(t, LandTile)
        }
        self.ports_by_id = {p.id: p for p in self.tiles.values() if isinstance(p, Port)}

        return self


def init_port_nodes_cache(
    tiles: Dict[Coordinate, Tile]
) -> Dict[Union[FastResource, None], Set[int]]:
    """Initializes board.port_nodes cache.

    Args:
        tiles (Dict[Coordinate, Tile]): initialized tiles datastructure

    Returns:
        Dict[Union[FastResource, None], Set[int]]: Mapping from FastResource to node_ids that
            enable port trading. None key represents 3:1 port.
    """
    port_nodes = defaultdict(set)
    for tile in tiles.values():
        if not isinstance(tile, Port):
            continue

        (a_noderef, b_noderef) = PORT_DIRECTION_TO_NODEREFS[tile.direction]
        port_nodes[tile.resource].add(tile.nodes[a_noderef])
        port_nodes[tile.resource].add(tile.nodes[b_noderef])
    return port_nodes


def init_adjacent_tiles(
    land_tiles: Dict[Coordinate, LandTile]
) -> Dict[int, List[LandTile]]:
    adjacent_tiles = defaultdict(list)  # node_id => tile[3]
    for tile in land_tiles.values():
        for node_id in tile.nodes.values():
            adjacent_tiles[node_id].append(tile)
    return adjacent_tiles


def init_node_production(
    adjacent_tiles: Dict[int, List[LandTile]]
) -> Dict[NodeId, Counter]:
    """Returns node_id => Counter({WHEAT: 0.123, ...})"""
    node_production = dict()
    for node_id in adjacent_tiles.keys():
        node_production[node_id] = get_node_counter_production(adjacent_tiles, node_id)
    return node_production


def get_node_counter_production(adjacent_tiles, node_id):
    tiles = adjacent_tiles[node_id]
    return Counter(
        {
            t.resource: number_probability(t.number)
            for t in tiles
            if t.resource is not None
        }
    )


def build_dice_probas():
    probas = defaultdict(float)
    for i in range(1, 7):
        for j in range(1, 7):
            probas[i + j] += 1 / 36
    return probas


DICE_PROBAS = build_dice_probas()


def number_probability(number):
    return DICE_PROBAS[number]


def initialize_tiles(
    map_template: MapTemplate,
    shuffled_numbers_param=None,
    shuffled_port_resources_param=None,
    shuffled_tile_resources_param=None,
) -> Dict[Coordinate, Tile]:
    """Initializes a new random board, based on the MapTemplate.

    It first shuffles tiles, ports, and numbers. Then goes satisfying the
    topology (i.e. placing tiles on coordinates); ensuring to "attach" these to
    neighbor tiles (so as to not repeat nodes or edges objects).

    Args:
        map_template (MapTemplate): Template to initialize.

    Raises:
        ValueError: Invalid tile in topology

    Returns:
        Dict[Coordinate, Tile]: Coordinate to initialized Tile mapping.
    """
    shuffled_port_resources = shuffled_port_resources_param or random.sample(
        map_template.port_resources, len(map_template.port_resources)
    )
    shuffled_tile_resources = shuffled_tile_resources_param or random.sample(
        map_template.tile_resources, len(map_template.tile_resources)
    )
    shuffled_numbers = shuffled_numbers_param or random.sample(
        map_template.numbers, len(map_template.numbers)
    )

    # for each topology entry, place a tile. keep track of nodes and edges
    all_tiles: Dict[Coordinate, Tile] = {}
    node_autoinc = 0
    tile_autoinc = 0
    port_autoinc = 0
    for coordinate, tile_type in map_template.topology.items():
        nodes, edges, node_autoinc = get_nodes_and_edges(
            all_tiles, coordinate, node_autoinc
        )

        # create and save tile
        if isinstance(tile_type, tuple):  # is port
            (_, direction) = tile_type
            port = Port(
                port_autoinc, shuffled_port_resources.pop(), direction, nodes, edges
            )
            all_tiles[coordinate] = port
            port_autoinc += 1
        elif tile_type == LandTile:
            resource = shuffled_tile_resources.pop()
            if resource != None:
                number = shuffled_numbers.pop()
                tile = LandTile(tile_autoinc, resource, number, nodes, edges)
            else:
                tile = LandTile(tile_autoinc, None, None, nodes, edges)  # desert
            all_tiles[coordinate] = tile
            tile_autoinc += 1
        elif tile_type == Water:
            water_tile = Water(nodes, edges)
            all_tiles[coordinate] = water_tile
        else:
            raise ValueError("Invalid tile")

    return all_tiles


def get_nodes_and_edges(tiles, coordinate: Coordinate, node_autoinc):
    """Get pre-existing nodes and edges in board for given tile coordinate"""
    nodes = {
        NodeRef.NORTH: None,
        NodeRef.NORTHEAST: None,
        NodeRef.SOUTHEAST: None,
        NodeRef.SOUTH: None,
        NodeRef.SOUTHWEST: None,
        NodeRef.NORTHWEST: None,
    }
    edges = {
        EdgeRef.EAST: None,
        EdgeRef.SOUTHEAST: None,
        EdgeRef.SOUTHWEST: None,
        EdgeRef.WEST: None,
        EdgeRef.NORTHWEST: None,
        EdgeRef.NORTHEAST: None,
    }

    # Find pre-existing ones
    neighbor_tiles = [(add(coordinate, UNIT_VECTORS[d]), d) for d in Direction]
    for coord, neighbor_direction in neighbor_tiles:
        if coord not in tiles:
            continue

        neighbor = tiles[coord]
        if neighbor_direction == Direction.EAST:
            nodes[NodeRef.NORTHEAST] = neighbor.nodes[NodeRef.NORTHWEST]
            nodes[NodeRef.SOUTHEAST] = neighbor.nodes[NodeRef.SOUTHWEST]
            edges[EdgeRef.EAST] = neighbor.edges[EdgeRef.WEST]
        elif neighbor_direction == Direction.SOUTHEAST:
            nodes[NodeRef.SOUTH] = neighbor.nodes[NodeRef.NORTHWEST]
            nodes[NodeRef.SOUTHEAST] = neighbor.nodes[NodeRef.NORTH]
            edges[EdgeRef.SOUTHEAST] = neighbor.edges[EdgeRef.NORTHWEST]
        elif neighbor_direction == Direction.SOUTHWEST:
            nodes[NodeRef.SOUTH] = neighbor.nodes[NodeRef.NORTHEAST]
            nodes[NodeRef.SOUTHWEST] = neighbor.nodes[NodeRef.NORTH]
            edges[EdgeRef.SOUTHWEST] = neighbor.edges[EdgeRef.NORTHEAST]
        elif neighbor_direction == Direction.WEST:
            nodes[NodeRef.NORTHWEST] = neighbor.nodes[NodeRef.NORTHEAST]
            nodes[NodeRef.SOUTHWEST] = neighbor.nodes[NodeRef.SOUTHEAST]
            edges[EdgeRef.WEST] = neighbor.edges[EdgeRef.EAST]
        elif neighbor_direction == Direction.NORTHWEST:
            nodes[NodeRef.NORTH] = neighbor.nodes[NodeRef.SOUTHEAST]
            nodes[NodeRef.NORTHWEST] = neighbor.nodes[NodeRef.SOUTH]
            edges[EdgeRef.NORTHWEST] = neighbor.edges[EdgeRef.SOUTHEAST]
        elif neighbor_direction == Direction.NORTHEAST:
            nodes[NodeRef.NORTH] = neighbor.nodes[NodeRef.SOUTHWEST]
            nodes[NodeRef.NORTHEAST] = neighbor.nodes[NodeRef.SOUTH]
            edges[EdgeRef.NORTHEAST] = neighbor.edges[EdgeRef.SOUTHWEST]
        else:
            raise Exception("Something went wrong")

    # Initializes new ones
    for noderef, value in nodes.items():
        if value is None:
            nodes[noderef] = node_autoinc
            node_autoinc += 1
    for edgeref, value in edges.items():
        if value is None:
            a_noderef, b_noderef = get_edge_nodes(edgeref)
            edge_nodes = (nodes[a_noderef], nodes[b_noderef])
            edges[edgeref] = edge_nodes  # type: ignore

    return (
        typing.cast(Dict[NodeRef, NodeId], nodes),
        typing.cast(Dict[EdgeRef, EdgeId], edges),
        node_autoinc,
    )


def get_edge_nodes(edge_ref):
    """returns pair of nodes at the "ends" of a given edge"""
    return {
        EdgeRef.EAST: (NodeRef.NORTHEAST, NodeRef.SOUTHEAST),
        EdgeRef.SOUTHEAST: (NodeRef.SOUTHEAST, NodeRef.SOUTH),
        EdgeRef.SOUTHWEST: (NodeRef.SOUTH, NodeRef.SOUTHWEST),
        EdgeRef.WEST: (NodeRef.SOUTHWEST, NodeRef.NORTHWEST),
        EdgeRef.NORTHWEST: (NodeRef.NORTHWEST, NodeRef.NORTH),
        EdgeRef.NORTHEAST: (NodeRef.NORTH, NodeRef.NORTHEAST),
    }[edge_ref]


# TODO: Could consolidate Direction with EdgeRef.
PORT_DIRECTION_TO_NODEREFS = {
    Direction.WEST: (NodeRef.NORTHWEST, NodeRef.SOUTHWEST),
    Direction.NORTHWEST: (NodeRef.NORTH, NodeRef.NORTHWEST),
    Direction.NORTHEAST: (NodeRef.NORTHEAST, NodeRef.NORTH),
    Direction.EAST: (NodeRef.SOUTHEAST, NodeRef.NORTHEAST),
    Direction.SOUTHEAST: (NodeRef.SOUTH, NodeRef.SOUTHEAST),
    Direction.SOUTHWEST: (NodeRef.SOUTHWEST, NodeRef.SOUTH),
}

TOURNAMENT_MAP_TILES = initialize_tiles(
    BASE_MAP_TEMPLATE,
    [10, 8, 3, 6, 2, 5, 10, 8, 4, 11, 12, 9, 5, 4, 9, 11, 3, 6],
    [
        None,
        SHEEP,
        None,
        ORE,
        WHEAT,
        None,
        WOOD,
        BRICK,
        None,
    ],
    [
        None,
        WOOD,
        SHEEP,
        SHEEP,
        WOOD,
        WHEAT,
        WOOD,
        WHEAT,
        BRICK,
        SHEEP,
        BRICK,
        SHEEP,
        WHEAT,
        WHEAT,
        ORE,
        BRICK,
        ORE,
        WOOD,
        ORE,
        None,
    ],
)
TOURNAMENT_MAP = CatanMap.from_tiles(TOURNAMENT_MAP_TILES)


def build_map(map_type: Literal["BASE", "TOURNAMENT", "MINI"]):
    if map_type == "TOURNAMENT":
        return TOURNAMENT_MAP  # this assumes map is read-only data struct
    elif map_type == "MINI":
        return CatanMap.from_template(MINI_MAP_TEMPLATE)
    else:
        return CatanMap.from_template(BASE_MAP_TEMPLATE)
