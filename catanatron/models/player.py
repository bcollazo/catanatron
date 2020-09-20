import random
from enum import Enum

from catanatron.models.decks import ResourceDeck


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Player:
    def __init__(self, color):
        self.color = color
        self.public_victory_points = 0
        self.actual_victory_points = 0
        self.resource_deck = ResourceDeck(empty=True)

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError

    def discard(self):
        """Must return n/2 cards to discard from self.resource_deck"""
        raise NotImplementedError

    def receive(self, resource_deck):
        self.resource_deck += resource_deck

    def has_knight_card(self):
        return False


class SimplePlayer(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]

    def discard(self):
        cards = self.resource_deck.to_array()
        return cards[: len(cards) // 2]


class RandomPlayer(Player):
    def decide(self, game, playable_actions):
        index = random.randrange(0, len(playable_actions))
        return playable_actions[index]

    def discard(self):
        cards = self.resource_deck.to_array()
        return random.sample(cards, len(cards) // 2)
