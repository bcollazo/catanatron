import random
from enum import Enum


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


class Tile:
    def __init__(self, resource=None):
        self.resource = resource
        self.desert = resource == None
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

NUMBERS_DECK = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 12]


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


generate_board()
