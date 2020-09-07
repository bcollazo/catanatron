import random
from enum import Enum


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Player:
    def __init__(self, color):
        self.color = color

    def decide(self, board, playable_actions):
        raise NotImplementedError

    def has_knight_card(self):
        return False


class RandomPlayer(Player):
    def decide(self, board, playable_actions):
        index = random.randrange(0, len(playable_actions))
        return playable_actions[index]
