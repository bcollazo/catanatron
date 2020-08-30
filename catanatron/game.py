from catanatron.models.map import BaseMap
from catanatron.models.board import Board


# TODO: This will contain the turn-by-turn controlling logic.
class Game:
    def __init__(self):
        # Map is defined by having a large-enough coordinate_system
        # and reading from a (coordinate) => Tile | Water | Port map
        # and filling the rest with water tiles.
        self.map = BaseMap()

        # board should be: (coordinate) => Tile (with nodes and edges initialized)
        self.board = Board(self.map)

        # TODO: Create players.
