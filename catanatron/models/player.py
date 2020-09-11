import random
from enum import Enum

from catanatron.models.decks import ResourceDecks


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Player:
    def __init__(self, color):
        self.color = color
        self.resource_decks = ResourceDecks(empty=True)

    def decide(self, game, playable_actions):
        """Must return one of the playable_actions.

        Args:
            game (Game): to use only in a read-only manner
            playable_actions ([type]): options right now
        """
        raise NotImplementedError

    def receive(self, resource_decks):
        self.resource_decks += resource_decks

    def has_knight_card(self):
        return False


class SimplePlayer(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]


class RandomPlayer(Player):
    def decide(self, game, playable_actions):
        index = random.randrange(0, len(playable_actions))
        return playable_actions[index]
