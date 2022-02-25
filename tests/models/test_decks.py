from catanatron.models.enums import (
    KNIGHT,
    ORE,
    SHEEP,
    VICTORY_POINT,
    BRICK,
    WHEAT,
    WOOD,
)
from catanatron.models.decks import (
    draw_from_listdeck,
    freqdeck_add,
    freqdeck_from_listdeck,
    freqdeck_replenish,
    freqdeck_can_draw,
    freqdeck_count,
    freqdeck_draw,
    freqdeck_replenish,
    freqdeck_subtract,
    starting_devcard_bank,
    starting_devcard_proba,
    starting_resource_bank,
)


def test_resource_freqdeck_init():
    deck = starting_resource_bank()
    assert deck[0] == 19


def test_resource_freqdeck_can_draw():
    deck = starting_resource_bank()
    assert freqdeck_can_draw(deck, 10, BRICK)
    assert not freqdeck_can_draw(deck, 20, BRICK)


def test_resource_freqdeck_integration():
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


def test_can_add():
    a = [0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0]

    freqdeck_replenish(a, 10, ORE)
    freqdeck_replenish(b, 1, ORE)

    assert freqdeck_count(a, ORE) == 10
    assert freqdeck_count(b, ORE) == 1
    b = freqdeck_add(b, a)
    assert freqdeck_count(a, ORE) == 10
    assert freqdeck_count(b, ORE) == 11


def test_can_subtract():
    a = [0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0]

    freqdeck_replenish(a, 13, SHEEP)
    freqdeck_replenish(b, 4, SHEEP)

    assert freqdeck_count(a, SHEEP) == 13
    assert freqdeck_count(b, SHEEP) == 4

    freqdeck_replenish(b, 11, SHEEP)  # now has 15
    b = freqdeck_subtract(b, a)
    assert freqdeck_count(a, SHEEP) == 13
    assert freqdeck_count(b, SHEEP) == 2


def test_from_array():
    a = freqdeck_from_listdeck([BRICK, BRICK, WOOD])
    assert sum(a) == 3
    assert freqdeck_count(a, BRICK) == 2
    assert freqdeck_count(a, WOOD) == 1


def test_deck_proba():
    assert starting_devcard_proba(KNIGHT) == 14 / 25
    assert starting_devcard_proba(VICTORY_POINT) == 5 / 25


def test_draw_from_listdeck():
    listdeck = [1, 2, 2, 4]
    draw_from_listdeck(listdeck, 1, 2)
    assert listdeck == [1, 2, 4]
