import pytest

from catanatron.models.enums import Resource
from catanatron.models.decks import ResourceDecks


def test_resource_deck_init():
    decks = ResourceDecks()
    assert decks.count(Resource.WOOD) == 19


def test_resource_deck_can_draw():
    decks = ResourceDecks()
    assert decks.can_draw(10, Resource.BRICK)
    assert not decks.can_draw(20, Resource.BRICK)


def test_resource_deck_integration():
    decks = ResourceDecks()
    assert decks.count(Resource.WHEAT) == 19

    assert decks.can_draw(10, Resource.WHEAT)
    decks.draw(10, Resource.WHEAT)
    assert decks.count(Resource.WHEAT) == 9

    with pytest.raises(ValueError):  # not enough
        decks.draw(10, Resource.WHEAT)

    decks.draw(9, Resource.WHEAT)
    assert decks.count(Resource.WHEAT) == 0

    with pytest.raises(ValueError):  # not enough
        decks.draw(1, Resource.WHEAT)

    decks.replenish(2, Resource.WHEAT)
    assert decks.count(Resource.WHEAT) == 2

    decks.draw(1, Resource.WHEAT)
    assert decks.count(Resource.WHEAT) == 1
