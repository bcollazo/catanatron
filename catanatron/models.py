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
