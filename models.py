import random
from enum import Enum


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


class Tile:
    def __init__(self, resource):
        self.resource = resource


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
]
