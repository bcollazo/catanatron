import timeit

setup = """
from catanatron.models.decks import Deck, ResourceDeck, DevelopmentDeck
from catanatron.models.enums import Resource
"""

code = """
deck = ResourceDeck.starting_bank()
assert deck.count(Resource.WOOD) == 19

deck = ResourceDeck.starting_bank()
assert deck.can_draw(10, Resource.BRICK)
assert not deck.can_draw(20, Resource.BRICK)

deck = ResourceDeck.starting_bank()
assert deck.count(Resource.WHEAT) == 19
assert deck.num_cards() == 19 * 5

assert deck.can_draw(10, Resource.WHEAT)
deck.draw(10, Resource.WHEAT)
assert deck.count(Resource.WHEAT) == 9

deck.draw(9, Resource.WHEAT)
assert deck.count(Resource.WHEAT) == 0

deck.replenish(2, Resource.WHEAT)
assert deck.count(Resource.WHEAT) == 2

deck.draw(1, Resource.WHEAT)
assert deck.count(Resource.WHEAT) == 1

a = ResourceDeck()
b = ResourceDeck()

a.replenish(10, Resource.ORE)
b.replenish(1, Resource.ORE)

assert a.count(Resource.ORE) == 10
assert b.count(Resource.ORE) == 1
b += a
assert a.count(Resource.ORE) == 10
assert b.count(Resource.ORE) == 11

a = ResourceDeck()
b = ResourceDeck()

a.replenish(13, Resource.SHEEP)
b.replenish(4, Resource.SHEEP)

assert a.count(Resource.SHEEP) == 13
assert b.count(Resource.SHEEP) == 4

b.replenish(11, Resource.SHEEP)  # now has 15
b -= a
assert a.count(Resource.SHEEP) == 13
assert b.count(Resource.SHEEP) == 2

a = ResourceDeck()
assert len(a.to_array()) == 0

a.replenish(3, Resource.SHEEP)
a.replenish(2, Resource.BRICK)
assert len(a.to_array()) == 5
assert len(set(a.to_array())) == 2

a = DevelopmentDeck.starting_bank()
num_cards = a.num_cards()

a.random_draw()

a = ResourceDeck.from_array(
    [
        Resource.BRICK,
        Resource.BRICK,
        Resource.WOOD,
    ]
)
assert a.num_cards() == 3
assert a.count(Resource.BRICK) == 2
assert a.count(Resource.WOOD) == 1
"""

NUMBER = 10000
result = timeit.timeit(
    code,
    setup=setup,
    number=NUMBER,
)
print("deck integration", result / NUMBER, "secs")

# Results:
# deck integration 7.86495955006103e-05 secs
