import random
from enum import Enum

from catanatron.models.decks import ResourceDeck, DevelopmentDeck
from catanatron.models.enums import DevelopmentCard


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
        self.resource_deck = ResourceDeck()
        self.development_deck = DevelopmentDeck()

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError

    def receive(self, resource_deck):
        self.resource_deck += resource_deck

    def has_knight_card(self):
        return self.development_deck.count(DevelopmentCard.KNIGHT) > 0

    def has_year_of_plenty_card(self):
        return self.development_deck.count(DevelopmentCard.YEAR_OF_PLENTY) > 0

    def has_monopoly_card(self):
        return self.development_deck.count(DevelopmentCard.MONOPOLY) > 0

    def has_road_building(self):
        return self.development_deck.count(DevelopmentCard.ROAD_BUILDING) > 0

    def __repr__(self):
        return type(self).__name__ + "[" + self.color.value + "]"


class SimplePlayer(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]


class RandomPlayer(Player):
    def decide(self, game, playable_actions):
        index = random.randrange(0, len(playable_actions))
        return playable_actions[index]
