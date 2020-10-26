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
    def __init__(self, color, name=None):
        self.name = name
        self.color = color
        self.public_victory_points = 0
        self.actual_victory_points = 0
        self.resource_deck = ResourceDeck()
        self.development_deck = DevelopmentDeck()

        self.clean_turn_state()

    def clean_turn_state(self):
        self.playable_development_cards = self.development_deck.to_array()
        self.has_rolled = False

    def mark_played_dev_card(self):
        self.playable_development_cards = []

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError

    def receive(self, resource_deck):
        self.resource_deck += resource_deck

    def can_play_knight(self):
        return DevelopmentCard.KNIGHT in self.playable_development_cards

    def can_play_year_of_plenty(self):
        return DevelopmentCard.YEAR_OF_PLENTY in self.playable_development_cards

    def can_play_monopoly(self):
        return DevelopmentCard.MONOPOLY in self.playable_development_cards

    def can_play_road_building(self):
        return DevelopmentCard.ROAD_BUILDING in self.playable_development_cards

    def __repr__(self):
        return f"{type(self).__name__}:{self.name}[{self.color.value}]"


class SimplePlayer(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]


class RandomPlayer(Player):
    def decide(self, game, playable_actions):
        index = random.randrange(0, len(playable_actions))
        return playable_actions[index]
