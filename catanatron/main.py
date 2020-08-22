import random
from models import Tile, Port, Resource, Board
from coordinate_system import generate_coordinate_system

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
    shuffled_ports = random.sample(PORT_DECK, len(PORT_DECK))
    shuffled_tiles = random.sample(TILE_DECK, len(TILE_DECK))
    shuffled_numbers = random.sample(NUMBERS_DECK, len(NUMBERS_DECK))

    coordinates = generate_coordinate_system(3)  # where last layer is water

    # walk spirally placing tiles. initializing them (attach so as to not repeat nodes and edges)
    # result should be: (coordinate) => Tile (with nodes and edges initialized)

    # walk spirally placing numbers (skipping dessert)
    # walk spirally setting ports (and directions)

    # assemble by placing board

    print(shuffled_ports)
    print(shuffled_tiles)
    print(shuffled_numbers)
    return Board(shuffled_ports, shuffled_tiles, shuffled_numbers)


generate_board()
