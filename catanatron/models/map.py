from enum import Enum

from catanatron.models.coordinate_system import generate_coordinate_system, Direction
from catanatron.models.enums import Resource


class Tile:
    def __init__(self, tile_id, resource, number, nodes, edges):
        self.id = tile_id

        self.resource = resource  # None means desert tile
        self.number = number
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Tile:" + str(self.resource)


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
