import timeit

setup = """
from catanatron.models.decks import (
    starting_resource_bank, freqdeck_count, freqdeck_draw, freqdeck_can_draw,
    freqdeck_replenish, freqdeck_subtract, freqdeck_add, starting_devcard_bank,
    freqdeck_from_listdeck)
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, KNIGHT
"""

code = """
deck = starting_resource_bank()
assert freqdeck_count(deck, WOOD) == 19

deck = starting_resource_bank()
assert freqdeck_can_draw(deck, 10, BRICK)
assert not freqdeck_can_draw(deck, 20, BRICK)

deck = starting_resource_bank()
assert freqdeck_count(deck, WHEAT) == 19
assert sum(deck) == 19 * 5

assert freqdeck_can_draw(deck, 10, WHEAT)
freqdeck_draw(deck, 10, WHEAT)
assert freqdeck_count(deck, WHEAT) == 9

freqdeck_draw(deck, 9, WHEAT)
assert freqdeck_count(deck, WHEAT) == 0

freqdeck_replenish(deck, 2, WHEAT)
assert freqdeck_count(deck, WHEAT) == 2

freqdeck_draw(deck, 1, WHEAT)
assert freqdeck_count(deck, WHEAT) == 1

a = [0,0,0,0,0]
b = [0,0,0,0,0]

freqdeck_replenish(a, 10, ORE)
freqdeck_replenish(b, 1, ORE)

assert freqdeck_count(a, ORE) == 10
assert freqdeck_count(b, ORE) == 1
b = freqdeck_add(b, a)
assert freqdeck_count(a, ORE) == 10
assert freqdeck_count(b, ORE) == 11

a = [0,0,0,0,0]
b = [0,0,0,0,0]

freqdeck_replenish(a, 13, SHEEP)
freqdeck_replenish(b, 4, SHEEP)

assert freqdeck_count(a, SHEEP) == 13
assert freqdeck_count(b, SHEEP) == 4

freqdeck_replenish(b, 11, SHEEP)  # now has 15
b = freqdeck_subtract(b, a)
assert freqdeck_count(a, SHEEP) == 13
assert freqdeck_count(b, SHEEP) == 2

a = [0,0,0,0,0]

freqdeck_replenish(a, 3, SHEEP)
freqdeck_replenish(a, 2, BRICK)

a = starting_devcard_bank()
num_cards = len(a)

a.pop()

a = freqdeck_from_listdeck(
    [
        BRICK,
        BRICK,
        WOOD,
    ]
)
assert sum(a) == 3
assert freqdeck_count(a, BRICK) == 2
assert freqdeck_count(a, WOOD) == 1
"""

NUMBER = 10000
result = timeit.timeit(
    code,
    setup=setup,
    number=NUMBER,
)
print("deck integration", result / NUMBER, "secs")

# Results:
# deck integration 4.865029100000001e-06 secs
