import pytest
from unittest.mock import MagicMock, patch

from catanatron.state import State, apply_action, yield_resources
from catanatron.models.map import BaseMap
from catanatron.models.board import Board
from catanatron.models.enums import Resource, DevelopmentCard, BuildingType
from catanatron.models.actions import ActionType, Action, ActionPrompt
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.decks import ResourceDeck


def test_buying_road_is_payed_for():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players, BaseMap())

    state.board.build_road = MagicMock()
    players[0].resource_deck = ResourceDeck()
    action = Action(players[0].color, ActionType.BUILD_ROAD, (3, 4))
    with pytest.raises(ValueError):  # not enough money
        apply_action(state, action)

    players[0].resource_deck = ResourceDeck.from_array([Resource.WOOD, Resource.BRICK])
    apply_action(state, action)

    assert players[0].resource_deck.count(Resource.WOOD) == 0
    assert players[0].resource_deck.count(Resource.BRICK) == 0


def test_moving_robber_steals_correctly():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players, BaseMap())

    players[1].resource_deck.replenish(1, Resource.WHEAT)
    state.board.build_settlement(Color.BLUE, 3, initial_build_phase=True)

    action = Action(players[0].color, ActionType.MOVE_ROBBER, ((2, 0, -2), None, None))
    apply_action(state, action)
    assert players[0].resource_deck.num_cards() == 0
    assert players[1].resource_deck.num_cards() == 1

    action = Action(
        players[0].color,
        ActionType.MOVE_ROBBER,
        ((0, 0, 0), players[1].color, Resource.WHEAT),
    )
    apply_action(state, action)
    assert players[0].resource_deck.num_cards() == 1
    assert players[1].resource_deck.num_cards() == 0


@patch("catanatron.state.roll_dice")
def test_seven_cards_dont_trigger_discarding(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players, BaseMap())
    blue_seating = state.players.index(players[1])

    players[1].resource_deck = ResourceDeck()
    players[1].resource_deck.replenish(7, Resource.WHEAT)
    apply_action(state, Action(players[0].color, ActionType.ROLL, None))  # roll

    discarding_ticks = list(
        filter(
            lambda a: a[0] == blue_seating and a[1] == ActionPrompt.DISCARD,
            state.tick_queue,
        )
    )
    assert len(discarding_ticks) == 0
