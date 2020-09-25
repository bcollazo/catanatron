import pytest

from catanatron.models.enums import Resource
from catanatron.models.decks import ResourceDeck, DevelopmentDeck


def test_resource_deck_init():
    deck = ResourceDeck.starting_bank()
    assert deck.count(Resource.WOOD) == 19


def test_resource_deck_can_draw():
    deck = ResourceDeck.starting_bank()
    assert deck.can_draw(10, Resource.BRICK)
    assert not deck.can_draw(20, Resource.BRICK)


def test_resource_deck_integration():
    deck = ResourceDeck.starting_bank()
    assert deck.count(Resource.WHEAT) == 19
    assert deck.num_cards() == 19 * 5

    assert deck.can_draw(10, Resource.WHEAT)
    deck.draw(10, Resource.WHEAT)
    assert deck.count(Resource.WHEAT) == 9

    with pytest.raises(ValueError):  # not enough
        deck.draw(10, Resource.WHEAT)

    deck.draw(9, Resource.WHEAT)
    assert deck.count(Resource.WHEAT) == 0

    with pytest.raises(ValueError):  # not enough
        deck.draw(1, Resource.WHEAT)

    deck.replenish(2, Resource.WHEAT)
    assert deck.count(Resource.WHEAT) == 2

    deck.draw(1, Resource.WHEAT)
    assert deck.count(Resource.WHEAT) == 1


def test_can_add():
    a = ResourceDeck()
    b = ResourceDeck()

    a.replenish(10, Resource.ORE)
    b.replenish(1, Resource.ORE)

    assert a.count(Resource.ORE) == 10
    assert b.count(Resource.ORE) == 1
    b += a
    assert a.count(Resource.ORE) == 10
    assert b.count(Resource.ORE) == 11


def test_can_subtract():
    a = ResourceDeck()
    b = ResourceDeck()

    a.replenish(13, Resource.SHEEP)
    b.replenish(4, Resource.SHEEP)

    assert a.count(Resource.SHEEP) == 13
    assert b.count(Resource.SHEEP) == 4
    with pytest.raises(ValueError):  # not enough
        b -= a

    b.replenish(11, Resource.SHEEP)  # now has 15
    b -= a
    assert a.count(Resource.SHEEP) == 13
    assert b.count(Resource.SHEEP) == 2


def test_to_array():
    a = ResourceDeck()
    assert len(a.to_array()) == 0

    a.replenish(3, Resource.SHEEP)
    a.replenish(2, Resource.BRICK)
    assert len(a.to_array()) == 5
    assert len(set(a.to_array())) == 2


def test_random_draw():
    a = DevelopmentDeck.starting_bank()
    num_cards = a.num_cards()

    a.random_draw()
    assert a.num_cards() == num_cards - 1
