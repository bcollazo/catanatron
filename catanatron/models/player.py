from collections import defaultdict
import random
from enum import Enum

from catanatron.models.decks import ResourceDeck, DevelopmentDeck
from catanatron.models.enums import DevelopmentCard, BuildingType


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Player:
    def __init__(self, color, name=None):
        self.name = name
        self.color = color

        self.restart_state()

    def restart_state(self):
        self.public_victory_points = 0
        self.actual_victory_points = 0

        self.resource_deck = ResourceDeck()
        self.development_deck = DevelopmentDeck()
        self.played_development_cards = DevelopmentDeck()

        self.has_road = False
        self.has_army = False

        self.roads_available = 15
        self.settlements_available = 5
        self.cities_available = 4

        self.has_rolled = False
        self.playable_development_cards = self.development_deck.to_array()

        self.buildings = defaultdict(list)  # dict of BuildingType => (node_id|edge)[]

    def build_settlement(self, node_id, is_free):
        self.buildings[BuildingType.SETTLEMENT].append(node_id)
        self.settlements_available -= 1
        if not is_free:
            self.resource_deck -= ResourceDeck.settlement_cost()

    def build_road(self, edge, is_free):
        self.buildings[BuildingType.ROAD].append(edge)
        self.roads_available -= 1
        if not is_free:
            self.resource_deck -= ResourceDeck.road_cost()

    def build_city(self, node_id):
        self.buildings[BuildingType.SETTLEMENT].remove(node_id)
        self.buildings[BuildingType.CITY].append(node_id)
        self.settlements_available += 1
        self.cities_available -= 1
        self.resource_deck -= ResourceDeck.city_cost()

    def clean_turn_state(self):
        self.has_rolled = False
        self.playable_development_cards = self.development_deck.to_array()

    def mark_played_dev_card(self, card_type):
        self.development_deck.draw(1, card_type)
        self.played_development_cards.replenish(1, card_type)
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


class HumanPlayer(Player):
    def decide(self, game, playable_actions):
        print(self.resource_deck.to_array())
        print(self.development_deck.to_array())
        for i, action in enumerate(playable_actions):
            print(f"{i}: {action.action_type} {action.value}")
        i = None
        while i is None or (i < 0 or i >= len(playable_actions)):
            print("Please enter a valid index:")
            try:
                x = input(">>> ")
                i = int(x)
            except ValueError:
                pass

        return playable_actions[i]


class RandomPlayer(Player):
    def decide(self, game, playable_actions):
        index = random.randrange(0, len(playable_actions))
        return playable_actions[index]
