import random
from enum import Enum


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


class Tile:
    def __init__(self, resource=None):
        self.resource = resource
        self.is_desert = resource == None

        self.intersections = []

    def __repr__(self):
        return "Tile:" + str(self.resource)


class Port:
    # No resource means its a 3:1 port.
    def __init__(self, resource=None):
        self.resource = resource

    def __repr__(self):
        return "Port:" + str(self.resource)


class Edge:
    def __init__(self):
        pass


class Node:
    def __init__(self):
        pass


class Board:
    def __init__(self, ports, tiles, numbers):
        self.ports = ports
        self.tiles = tiles
        self.numbers = numbers


TILE_DECK = [
    # Four wood tiles
    Tile(Resource.WOOD),
    Tile(Resource.WOOD),
    Tile(Resource.WOOD),
    Tile(Resource.WOOD),
    # Three brick tiles
    Tile(Resource.BRICK),
    Tile(Resource.BRICK),
    Tile(Resource.BRICK),
    # Four sheep tiles
    Tile(Resource.SHEEP),
    Tile(Resource.SHEEP),
    Tile(Resource.SHEEP),
    Tile(Resource.SHEEP),
    # Four wheat tiles
    Tile(Resource.WHEAT),
    Tile(Resource.WHEAT),
    Tile(Resource.WHEAT),
    Tile(Resource.WHEAT),
    # Three ore tiles
    Tile(Resource.ORE),
    Tile(Resource.ORE),
    Tile(Resource.ORE),
    # Desert Tile
    Tile(),
]

PORT_DECK = [
    Port(Resource.WOOD),
    Port(Resource.BRICK),
    Port(Resource.SHEEP),
    Port(Resource.WHEAT),
    Port(Resource.ORE),
    Port(),
    Port(),
    Port(),
    Port(),
]

NUMBERS_DECK = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]


def generate_board():
    # Shuffle deck of ports
    shuffled_ports = random.sample(PORT_DECK, len(PORT_DECK))
    # Shuffle deck of tiles
    shuffled_tiles = random.sample(TILE_DECK, len(TILE_DECK))
    # Shuffle number circles on top of tiles
    shuffled_numbers = random.sample(NUMBERS_DECK, len(NUMBERS_DECK))

    print(shuffled_ports)
    print(shuffled_tiles)
    print(shuffled_numbers)
    return Board(shuffled_ports, shuffled_tiles, shuffled_numbers)


# We'll be using Cube coordinates in https://math.stackexchange.com/questions/2254655/hexagon-grid-coordinate-system
class Direction(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


def add(acoord, bcoord):
    (x, y, z) = acoord
    (u, v, w) = bcoord
    return (x + u, y + v, z + w)


UNIT_VECTORS = {
    # X-axis
    Direction.NORTHEAST: (1, 0, -1),
    Direction.SOUTHWEST: (-1, 0, 1),
    # Y-axis
    Direction.NORTHWEST: (0, 1, -1),
    Direction.SOUTHEAST: (0, -1, 1),
    # Z-axis
    Direction.EAST: (1, -1, 0),
    Direction.WEST: (-1, 1, 0),
}


def num_tiles_for(layer):
    """Including inner-layer tiles"""
    if layer == 0:
        return 1

    return 6 * layer + num_tiles_for(layer - 1)


def generate_coordinate_system(num_layers):
    """
    Generates a set of coordinates by expanding outward from a center tile on 
    (0,0,0) with the given number of layers (as in an onion :)). Follows BFS.
    """
    num_tiles = num_tiles_for(num_layers)

    agenda = [(0, 0, 0)]
    visited = set()
    while len(visited) < num_tiles:
        node = agenda.pop(0)
        visited.add(node)

        neighbors = [add(node, UNIT_VECTORS[d]) for d in Direction]
        new_neighbors = filter(
            lambda x: x not in visited and x not in agenda, neighbors
        )
        agenda.extend(new_neighbors)
    return visited
