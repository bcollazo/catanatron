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


def test_trade_execution():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    state = State(players, BaseMap())

    players[0].resource_deck.replenish(4, Resource.BRICK)
    trade_offer = tuple([Resource.BRICK] * 4 + [Resource.ORE])
    action = Action(players[0].color, ActionType.MARITIME_TRADE, trade_offer)
    apply_action(state, action)

    assert players[0].resource_deck.num_cards() == 1
    assert state.resource_deck.num_cards() == 19 * 5 + 4 - 1


# ===== Development Cards
def test_cant_buy_more_than_max_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players, BaseMap())

    with pytest.raises(ValueError):  # not enough money
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )

    players[0].resource_deck.replenish(26, Resource.SHEEP)
    players[0].resource_deck.replenish(26, Resource.WHEAT)
    players[0].resource_deck.replenish(26, Resource.ORE)

    for i in range(25):
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )

    # assert must have all victory points
    assert players[0].development_deck.num_cards() == 25
    assert players[0].development_deck.count(DevelopmentCard.VICTORY_POINT) == 5

    with pytest.raises(ValueError):  # not enough cards in bank
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )

    assert players[0].resource_deck.num_cards() == 3


def test_play_year_of_plenty_gives_player_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players, BaseMap())

    player_to_act = players[0]
    player_to_act.development_deck.replenish(1, DevelopmentCard.YEAR_OF_PLENTY)

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act.color,
        ActionType.PLAY_YEAR_OF_PLENTY,
        [Resource.ORE, Resource.WHEAT],
    )

    apply_action(state, action_to_execute)

    for card_type in Resource:
        if card_type == Resource.ORE or card_type == Resource.WHEAT:
            assert player_to_act.resource_deck.count(card_type) == 1
            assert state.resource_deck.count(card_type) == 18
        else:
            assert player_to_act.resource_deck.count(card_type) == 0
            assert state.resource_deck.count(card_type) == 19
    assert player_to_act.development_deck.count(DevelopmentCard.YEAR_OF_PLENTY) == 0


def test_play_monopoly_player_steals_cards():
    player_to_act = SimplePlayer(Color.RED)
    player_to_steal_from_1 = SimplePlayer(Color.BLUE)
    player_to_steal_from_2 = SimplePlayer(Color.ORANGE)

    player_to_act.development_deck.replenish(1, DevelopmentCard.MONOPOLY)

    player_to_steal_from_1.resource_deck.replenish(3, Resource.ORE)
    player_to_steal_from_1.resource_deck.replenish(1, Resource.WHEAT)
    player_to_steal_from_2.resource_deck.replenish(2, Resource.ORE)
    player_to_steal_from_2.resource_deck.replenish(1, Resource.WHEAT)

    players = [player_to_act, player_to_steal_from_1, player_to_steal_from_2]
    state = State(players, BaseMap())

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act.color, ActionType.PLAY_MONOPOLY, Resource.ORE
    )

    apply_action(state, action_to_execute)

    assert player_to_act.resource_deck.count(Resource.ORE) == 5
    assert player_to_steal_from_1.resource_deck.count(Resource.ORE) == 0
    assert player_to_steal_from_1.resource_deck.count(Resource.WHEAT) == 1
    assert player_to_steal_from_2.resource_deck.count(Resource.ORE) == 0
    assert player_to_steal_from_2.resource_deck.count(Resource.WHEAT) == 1


def test_can_only_play_one_dev_card_per_turn():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    state = State(players, BaseMap())

    players[0].development_deck.replenish(2, DevelopmentCard.YEAR_OF_PLENTY)

    players[0].clean_turn_state()
    action = Action(
        players[0].color, ActionType.PLAY_YEAR_OF_PLENTY, 2 * [Resource.BRICK]
    )
    apply_action(state, action)
    with pytest.raises(ValueError):  # shouldnt be able to play two dev cards
        apply_action(state, action)
