import random
from enum import Enum

from catanatron.coordinate_system import (
    generate_coordinate_system,
    Direction,
    add,
    UNIT_VECTORS,
)


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


# Given a tile, the reference to the node.
class NodeRef(Enum):
    NORTH = "NORTH"
    NORTHEAST = "NORTHEAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTH = "SOUTH"
    SOUTHWEST = "SOUTHWEST"
    NORTHWEST = "NORTHWEST"


class EdgeRef(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


class Tile:
    def __init__(self, resource, number, nodes, edges):
        self.is_desert = resource == None

        self.resource = resource
        self.number = number
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Tile:" + str(self.resource)


class Port:
    def __init__(self, resource, direction, nodes, edges):
        self.resource = resource  # No resource means its a 3:1 port.
        self.direction = direction
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Port:" + str(self.resource)


class Water:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


class Edge:
    next_autoinc_id = 0

    def __init__(self):
        self.id = Edge.next_autoinc_id
        Edge.next_autoinc_id += 1

    def __repr__(self):
        return "Edge:" + str(self.id)


class Node:
    next_autoinc_id = 0

    def __init__(self):
        self.id = Node.next_autoinc_id
        Node.next_autoinc_id += 1

    def __repr__(self):
        return "Node:" + str(self.id)


class BaseMap:
    """
    Describes a basic 4 player map. Includes the tiles, ports, and numbers used.
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
            (1, -1, 1): Tile,
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


class Game:
    def __init__(self):
        # Map is defined by having a large-enough coordinate_system
        # and reading from a (coordinate) => Tile | Water | Port map
        # and filling the rest with water tiles.
        self.map = BaseMap()

        # Initialization goes like: shuffle tiles, ports, and numbers.
        # Goes one by one placing tiles in board. When placing ensure they
        # are "attach" to possible neighbors. (no repeated nodes or edges)
        shuffled_port_resources = random.sample(
            self.map.port_resources, len(self.map.port_resources)
        )
        shuffled_tile_resources = random.sample(
            self.map.tile_resources, len(self.map.tile_resources)
        )
        shuffled_numbers = random.sample(self.map.numbers, len(self.map.numbers))

        # for each topology entry, place a tile.
        board = {}
        for (coordinate, tile_type) in self.map.topology.items():
            print(coordinate, tile_type)

            nodes, edges = get_nodes_and_edges(board, coordinate)
            if isinstance(tile_type, tuple):  # is port
                (TileClass, direction) = tile_type
                port = TileClass(shuffled_port_resources.pop(), direction, nodes, edges)
                board[coordinate] = port
            elif tile_type == Tile:
                resource = shuffled_tile_resources.pop()
                if resource != None:
                    number = shuffled_numbers.pop()
                    tile = Tile(resource, number, nodes, edges)
                else:
                    tile = Tile(None, None, nodes, edges)  # desert
                board[coordinate] = tile
            elif tile_type == Water:
                water_tile = Water(nodes, edges)
                board[coordinate] = water_tile
            else:
                raise Exception("Something went wrong")

        # board should be: (coordinate) => Tile (with nodes and edges initialized)
        self.board = board


def get_nodes_and_edges(board, coordinate):
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
    neighbors = [(add(coordinate, UNIT_VECTORS[d]), d) for d in Direction]
    for (coord, neighbor_direction) in neighbors:
        if coord not in board:
            continue

        neighbor = board[coord]
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
    for key, value in nodes.items():
        if value == None:
            nodes[key] = Node()
    for key, value in edges.items():
        if value == None:
            edges[key] = Edge()

    return nodes, edges
