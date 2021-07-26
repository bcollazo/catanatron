import random
import array

from catanatron.models.enums import Resource, DevelopmentCard


class Deck:
    @classmethod
    def from_array(cls, card_array):
        deck = cls()
        for card in card_array:
            deck.replenish(1, card)
        return deck

    def __init__(self, card_types):
        """Provides functionality to manage a pack of cards.

        Args:
            card_types (Enum): Describes cards to use
        """
        self.array = array.array("H", [0, 0, 0, 0, 0])
        self.indices = {c: i for i, c in enumerate(card_types)}

    def includes(self, other):
        return all([a >= b for a, b in zip(self.array, other.array)])

    def count(self, card_type):
        return self.array[self.indices[card_type]]

    def num_cards(self):
        return sum(self.array)

    def can_draw(self, count: int, card_type):
        return self.array[self.indices[card_type]] >= count

    def draw(self, count: int, card_type):
        if not self.can_draw(count, card_type):
            raise ValueError(f"Cant draw {count} {card_type}. Not enough cards.")

        self.array[self.indices[card_type]] -= count

    def random_draw(self):
        array = self.to_array()
        if len(array) == 0:
            raise ValueError(f"Cant random_draw. Not enough cards.")

        card_type = random.choice(array)
        self.draw(1, card_type)
        return card_type

    def replenish(self, count: int, card_type):
        self.array[self.indices[card_type]] += count

    def to_array(self):
        """Make it look like a deck of cards"""
        array = []
        for i, c in self.indices.items():
            array.extend([i] * self.array[c])
        return array

    def __add__(self, other):
        for i, c in enumerate(other.array):
            self.array[i] += c
        return self

    def __sub__(self, other):
        if not self.includes(other):
            raise ValueError("Invalid deck subtraction")
        for i, c in enumerate(other.array):
            self.array[i] -= c
        return self

    def __str__(self):
        return str(self.array)


class ResourceDeck(Deck):
    @staticmethod
    def starting_bank():
        deck = ResourceDeck()
        deck.replenish(19, Resource.WOOD)
        deck.replenish(19, Resource.BRICK)
        deck.replenish(19, Resource.SHEEP)
        deck.replenish(19, Resource.WHEAT)
        deck.replenish(19, Resource.ORE)
        return deck

    @staticmethod
    def road_cost():
        return ROAD_COST

    @staticmethod
    def settlement_cost():
        return SETTLEMENT_COST

    @staticmethod
    def city_cost():
        return CITY_COST

    @staticmethod
    def development_card_cost():
        return DEVELOPMENT_CARD_COST

    def __init__(self):
        Deck.__init__(self, Resource)


class DevelopmentDeck(Deck):
    @staticmethod
    def starting_card_proba(card):
        starting_deck = DevelopmentDeck.starting_bank()
        return starting_deck.count(card) / starting_deck.num_cards()

    @staticmethod
    def starting_bank():
        deck = DevelopmentDeck()
        deck.replenish(14, DevelopmentCard.KNIGHT)
        deck.replenish(2, DevelopmentCard.YEAR_OF_PLENTY)
        deck.replenish(2, DevelopmentCard.ROAD_BUILDING)
        deck.replenish(2, DevelopmentCard.MONOPOLY)
        deck.replenish(5, DevelopmentCard.VICTORY_POINT)
        return deck

    def __init__(self):
        Deck.__init__(self, DevelopmentCard)


ROAD_COST = ResourceDeck()
ROAD_COST.array = array.array("H", [1, 1, 0, 0, 0])

SETTLEMENT_COST = ResourceDeck()
SETTLEMENT_COST.array = array.array("H", [1, 1, 1, 1, 0])

CITY_COST = ResourceDeck()
CITY_COST.array = array.array("H", [0, 0, 0, 2, 3])

DEVELOPMENT_CARD_COST = ResourceDeck()
DEVELOPMENT_CARD_COST.array = array.array("H", [0, 0, 1, 1, 1])
