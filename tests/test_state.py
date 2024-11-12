import pytest

from catanatron.state import State, apply_action
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    player_clean_turn,
    player_freqdeck_add,
    player_deck_replenish,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.enums import (
    RESOURCES,
    ActionPrompt,
    BRICK,
    MONOPOLY,
    ORE,
    ActionType,
    Action,
    SHEEP,
    VICTORY_POINT,
    WHEAT,
    WOOD,
    YEAR_OF_PLENTY,
)
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.decks import freqdeck_count, freqdeck_from_listdeck


def test_buying_road_is_payed_for():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    state.is_initial_build_phase = False
    state.board.build_settlement(players[0].color, 3, True)
    action = Action(players[0].color, ActionType.BUILD_ROAD, (3, 4))
    player_freqdeck_add(
        state,
        players[0].color,
        freqdeck_from_listdeck([WOOD, BRICK]),
    )
    apply_action(state, action)

    assert player_num_resource_cards(state, players[0].color, WOOD) == 0
    assert player_num_resource_cards(state, players[0].color, BRICK) == 0


def test_moving_robber_steals_correctly():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    player_deck_replenish(state, players[1].color, WHEAT, 1)
    state.board.build_settlement(Color.BLUE, 3, initial_build_phase=True)

    action = Action(players[0].color, ActionType.MOVE_ROBBER, ((2, 0, -2), None, None))
    apply_action(state, action)
    assert player_num_resource_cards(state, players[0].color) == 0
    assert player_num_resource_cards(state, players[1].color) == 1

    action = Action(
        players[0].color,
        ActionType.MOVE_ROBBER,
        ((0, 0, 0), players[1].color, WHEAT),
    )
    apply_action(state, action)
    assert player_num_resource_cards(state, players[0].color) == 1
    assert player_num_resource_cards(state, players[1].color) == 0


def test_trade_execution():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    state = State(players)

    player_deck_replenish(state, players[0].color, BRICK, 4)
    trade_offer = tuple([BRICK] * 4 + [ORE])
    action = Action(players[0].color, ActionType.MARITIME_TRADE, trade_offer)
    apply_action(state, action)

    assert player_num_resource_cards(state, players[0].color) == 1
    assert sum(state.resource_freqdeck) == 19 * 5 + 4 - 1


# ===== Development Cards
def test_cant_buy_more_than_max_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    with pytest.raises(ValueError):  # not enough money
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )

    player_deck_replenish(state, players[0].color, SHEEP, 26)
    player_deck_replenish(state, players[0].color, WHEAT, 26)
    player_deck_replenish(state, players[0].color, ORE, 26)

    for i in range(25):
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )

    # assert must have all victory points
    assert player_num_dev_cards(state, players[0].color) == 25
    assert get_dev_cards_in_hand(state, players[0].color, VICTORY_POINT) == 5

    with pytest.raises(ValueError):  # not enough cards in bank
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )

    assert player_num_resource_cards(state, players[0].color) == 3


def test_play_year_of_plenty_gives_player_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    player_to_act = players[0]
    player_deck_replenish(state, player_to_act.color, YEAR_OF_PLENTY, 1)

    # Simulate end of turn which updates the OWNED_AT_START flags
    player_clean_turn(state, player_to_act.color)

    # Now try to play the card (as if it's next turn)
    action_to_execute = Action(
        player_to_act.color, ActionType.PLAY_YEAR_OF_PLENTY, [ORE, WHEAT]
    )

    apply_action(state, action_to_execute)

    for card_type in RESOURCES:
        if card_type == ORE or card_type == WHEAT:
            assert player_num_resource_cards(state, player_to_act.color, card_type) == 1
            assert freqdeck_count(state.resource_freqdeck, card_type) == 18
        else:
            assert player_num_resource_cards(state, player_to_act.color, card_type) == 0
            assert freqdeck_count(state.resource_freqdeck, card_type) == 19
    assert get_dev_cards_in_hand(state, player_to_act.color, YEAR_OF_PLENTY) == 0


def test_play_monopoly_player_steals_cards():
    player_to_act = SimplePlayer(Color.RED)
    player_to_steal_from_1 = SimplePlayer(Color.BLUE)
    player_to_steal_from_2 = SimplePlayer(Color.ORANGE)
    players = [player_to_act, player_to_steal_from_1, player_to_steal_from_2]
    state = State(players)

    player_deck_replenish(state, player_to_act.color, MONOPOLY)
    player_deck_replenish(state, player_to_steal_from_1.color, ORE, 3)
    player_deck_replenish(state, player_to_steal_from_1.color, WHEAT, 1)
    player_deck_replenish(state, player_to_steal_from_2.color, ORE, 2)
    player_deck_replenish(state, player_to_steal_from_2.color, WHEAT, 1)

    # Simulate end of turn which updates the OWNED_AT_START flags
    player_clean_turn(state, player_to_act.color)

    # Now try to play the card (as if it's next turn)
    action_to_execute = Action(player_to_act.color, ActionType.PLAY_MONOPOLY, ORE)
    apply_action(state, action_to_execute)

    assert player_num_resource_cards(state, player_to_act.color, ORE) == 5
    assert player_num_resource_cards(state, player_to_steal_from_1.color, ORE) == 0
    assert player_num_resource_cards(state, player_to_steal_from_1.color, WHEAT) == 1
    assert player_num_resource_cards(state, player_to_steal_from_2.color, ORE) == 0
    assert player_num_resource_cards(state, player_to_steal_from_2.color, WHEAT) == 1


def test_can_only_play_one_dev_card_per_turn():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    state = State(players)

    player_to_act = players[0]
    player_deck_replenish(state, player_to_act.color, YEAR_OF_PLENTY, 2)

    # Simulate end of turn which updates the OWNED_AT_START flags
    player_clean_turn(state, player_to_act.color)

    # Now try to play the card (as if it's next turn)
    action = Action(player_to_act.color, ActionType.PLAY_YEAR_OF_PLENTY, 2 * [BRICK])
    apply_action(state, action)
    with pytest.raises(ValueError):  # shouldnt be able to play two dev cards
        apply_action(state, action)


def test_sequence():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    state = State(players)

    p0_color = state.colors[0]
    assert state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
    assert Action(p0_color, ActionType.BUILD_SETTLEMENT, 0) in state.playable_actions
    assert Action(p0_color, ActionType.BUILD_SETTLEMENT, 50) in state.playable_actions

    apply_action(state, state.playable_actions[0])
