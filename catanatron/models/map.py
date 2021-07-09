import random
from enum import Enum
from collections import Counter, defaultdict
from typing import Dict, Set, Tuple

from catanatron.models.coordinate_system import Direction, add, UNIT_VECTORS
from catanatron.models.coordinate_system import generate_coordinate_system, Direction
from catanatron.models.enums import Resource

NUM_NODES = 54
NUM_EDGES = 72
NUM_TILES = 19


class Tile:
    def __init__(self, tile_id, resource, number, nodes, edges):
        self.id = tile_id

        self.resource = resource  # None means desert tile
        self.number = number

        self.nodes = nodes  # node_ref => node_id
        self.edges = edges  # edge_ref => edge

    def __repr__(self):
        if self.resource is None:
            return "Tile:Desert"
        return f"Tile:{self.number}{self.resource.value}"


class Port:
    def __init__(self, port_id, resource, direction, nodes, edges):
        self.id = port_id

        self.resource = resource  # None means its a 3:1 port.
        self.direction = direction
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Port:" + str(self.resource)


class Water:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


def init_port_nodes(catan_map) -> Dict[Resource, Set[int]]:
    """Initializes board.port_nodes cache.

    Args:
        catan_map (BaseMap): board instance.

    Returns:
        Dict[Union[Resource, None], Set[int]]: Mapping from Resource to node_ids that
            enable port trading. None key represents 3:1 port.
    """
    port_nodes = defaultdict(set)
    for (coordinate, value) in catan_map.topology.items():
        if not isinstance(value, tuple):
            continue

        tile = catan_map.tiles[coordinate]
        _, direction = value
        (a_noderef, b_noderef) = PORT_DIRECTION_TO_NODEREFS[direction]

        port_nodes[tile.resource].add(tile.nodes[a_noderef])
        port_nodes[tile.resource].add(tile.nodes[b_noderef])
    return port_nodes


def init_resource_tiles(catan_map):
    tiles = []
    for (coordinate, tile) in catan_map.tiles.items():
        if isinstance(tile, Port) or isinstance(tile, Water):
            continue
        tiles.append((coordinate, tile))
    return tiles


def init_adjacent_tiles(resource_tiles):
    adjacent_tiles = defaultdict(list)  # node_id => tile[3]
    for _, tile in resource_tiles:
        for node_id in tile.nodes.values():
            adjacent_tiles[node_id].append(tile)
    return adjacent_tiles


class BaseMap:
    """
    Represents a basic 4 player map.

    Holds both, template of map (self.topology) and the initialization of
    tiles, ports, and numbers (self.tiles) at random.
    """

    def __init__(self):
        self.numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        self.port_resources = [
            # These are 2:1 ports
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
            # These represet 3:1 ports
            None,
            None,
            None,
            None,
        ]
        self.tile_resources = [
            # Four wood tiles
            Resource.WOOD,
            Resource.WOOD,
            Resource.WOOD,
            Resource.WOOD,
            # Three brick tiles
            Resource.BRICK,
            Resource.BRICK,
            Resource.BRICK,
            # Four sheep tiles
            Resource.SHEEP,
            Resource.SHEEP,
            Resource.SHEEP,
            Resource.SHEEP,
            # Four wheat tiles
            Resource.WHEAT,
            Resource.WHEAT,
            Resource.WHEAT,
            Resource.WHEAT,
            # Three ore tiles
            Resource.ORE,
            Resource.ORE,
            Resource.ORE,
            # One desert
            None,
        ]

        # 3 layers, where last layer is water
        self.coordinate_system = generate_coordinate_system(3)
        self.topology = {
            # center
            (0, 0, 0): Tile,
            # first layer
            (1, -1, 0): Tile,
            (0, -1, 1): Tile,
            (-1, 0, 1): Tile,
            (-1, 1, 0): Tile,
            (0, 1, -1): Tile,
            (1, 0, -1): Tile,
            # second layer
            (2, -2, 0): Tile,
            (1, -2, 1): Tile,
            (0, -2, 2): Tile,
            (-1, -1, 2): Tile,
            (-2, 0, 2): Tile,
            (-2, 1, 1): Tile,
            (-2, 2, 0): Tile,
            (-1, 2, -1): Tile,
            (0, 2, -2): Tile,
            (1, 1, -2): Tile,
            (2, 0, -2): Tile,
            (2, -1, -1): Tile,
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
        }

        # (coordinate) => Tile (with nodes and edges initialized)
        self.tiles = initialize_board(self)

        # initialize auxiliary data structures for fast-lookups
        self.port_nodes = init_port_nodes(self)
        self.resource_tiles = init_resource_tiles(self)
        self.adjacent_tiles = init_adjacent_tiles(self.resource_tiles)
        self.node_production = init_node_production(
            self.adjacent_tiles
        )  # node-id => Counter
        self.tiles_by_id = {t.id: t for t in self.tiles.values() if isinstance(t, Tile)}
        self.ports_by_id = {p.id: p for p in self.tiles.values() if isinstance(p, Port)}


def init_node_production(adjacent_tiles):
    node_production = dict()
    for node_id in range(NUM_NODES):
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
    probas = defaultdict(int)
    for i in range(1, 7):
        for j in range(1, 7):
            probas[i + j] += 1 / 36
    return probas


DICE_PROBAS = build_dice_probas()


def number_probability(number):
    return DICE_PROBAS[number]


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


def initialize_board(catan_map) -> Dict[Tuple[int, int, int], Tile]:
    """Initializes a new random board, based on the catan_map template.

    It first shuffles tiles, ports, and numbers. Then goes satisfying the
    topology (i.e. placing tiles on coordinates); ensuring to "attach" these to
    neighbor tiles (so as to not repeat nodes or edges objects).

    Args:
        catan_map (BaseMap): Instance to mutate.

    Raises:
        ValueError: Invalid tile in topology

    Returns:
        Dict[tuple[int, int, int], Tile]: Coordinate to initialized Tile mapping.
    """
    shuffled_port_resources = random.sample(
        catan_map.port_resources, len(catan_map.port_resources)
    )
    shuffled_tile_resources = random.sample(
        catan_map.tile_resources, len(catan_map.tile_resources)
    )
    shuffled_numbers = random.sample(catan_map.numbers, len(catan_map.numbers))

    # for each topology entry, place a tile. keep track of nodes and edges
    all_tiles = {}
    node_autoinc = 0
    tile_autoinc = 0
    port_autoinc = 0
    for (coordinate, tile_type) in catan_map.topology.items():
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
        elif tile_type == Tile:
            resource = shuffled_tile_resources.pop()
            if resource != None:
                number = shuffled_numbers.pop()
                tile = Tile(tile_autoinc, resource, number, nodes, edges)
            else:
                tile = Tile(tile_autoinc, None, None, nodes, edges)  # desert
            all_tiles[coordinate] = tile
            tile_autoinc += 1
        elif tile_type == Water:
            water_tile = Water(nodes, edges)
            all_tiles[coordinate] = water_tile
        else:
            raise ValueError("Invalid tile")

    return all_tiles


def get_nodes_and_edges(tiles, coordinate, node_autoinc):
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
    for (coord, neighbor_direction) in neighbor_tiles:
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
            edges[edgeref] = edge_nodes

    return nodes, edges, node_autoinc


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
