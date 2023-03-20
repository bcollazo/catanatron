import pytest
from unittest.mock import MagicMock, patch

from catanatron.state_functions import (
    get_actual_victory_points,
    get_player_freqdeck,
    player_has_rolled,
)
from catanatron.game import Game, is_valid_trade
from catanatron.state import (
    apply_action,
    player_deck_replenish,
    player_num_resource_cards,
)
from catanatron.state_functions import player_key
from catanatron.models.actions import (
    generate_playable_actions
)
from catanatron.models.enums import (
    BRICK,
    ORE,
    RESOURCES,
    ActionPrompt,
    SETTLEMENT,
    ActionType,
    Action,
    WHEAT,
    WOOD,
    YEAR_OF_PLENTY,
    ROAD_BUILDING,
)
from catanatron.models.player import Color, RandomPlayer, SimplePlayer


def test_initial_build_phase():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    actions = []
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        actions.append(game.play_tick())

    p0_color = game.state.colors[0]
    assert (
        actions[0].action_type == ActionType.BUILD_SETTLEMENT
        and actions[0].color == p0_color
    )
    assert (
        actions[1].action_type == ActionType.BUILD_ROAD and actions[1].color == p0_color
    )
    assert (
        actions[2].action_type == ActionType.BUILD_SETTLEMENT
        and actions[2].color != p0_color
    )
    assert (
        actions[3].action_type == ActionType.BUILD_ROAD and actions[3].color != p0_color
    )
    assert (
        actions[4].action_type == ActionType.BUILD_SETTLEMENT
        and actions[4].color != p0_color
    )
    assert (
        actions[5].action_type == ActionType.BUILD_ROAD and actions[5].color != p0_color
    )
    assert (
        actions[6].action_type == ActionType.BUILD_SETTLEMENT
        and actions[6].color == p0_color
    )
    assert (
        actions[7].action_type == ActionType.BUILD_ROAD and actions[7].color == p0_color
    )
    assert (
        game.state.current_prompt == ActionPrompt.PLAY_TURN
        and game.state.current_color() == p0_color
    )

    assert game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 2
    assert game.state.player_state["P1_ACTUAL_VICTORY_POINTS"] == 2
    assert game.state.player_state["P0_VICTORY_POINTS"] == 2
    assert game.state.player_state["P1_VICTORY_POINTS"] == 2

    # assert there are 4 houses and 4 roads
    settlements = [
        building
        for building in game.state.board.buildings.values()
        if building[1] == SETTLEMENT
    ]
    assert len(settlements) == 4

    # assert should be house-road pairs, or together
    paths = game.state.board.continuous_roads_by_player(players[0].color)
    assert len(paths) == 1 or (
        len(paths) == 2 and len(paths[0]) == 1 and len(paths[1]) == 1
    )

    # assert should have resources from last house.
    # can only assert <= 3 b.c. player might place on a corner desert
    assert player_num_resource_cards(game.state, players[0].color) <= 3
    assert player_num_resource_cards(game.state, players[1].color) <= 3


def test_can_play_for_a_bit():  # assert no exception thrown
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    for _ in range(10):
        game.play_tick()


@patch("catanatron.state.roll_dice")
def test_seven_cards_dont_trigger_discarding(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]

    # Play initial build phase
    game = Game(players)
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    until_seven = 7 - player_num_resource_cards(game.state, players[1].color)
    player_deck_replenish(game.state, players[1].color, WHEAT, until_seven)
    assert player_num_resource_cards(game.state, players[1].color) == 7
    game.play_tick()  # should be player 0 rolling.

    assert not any(
        a.action_type == ActionType.DISCARD for a in game.state.playable_actions
    )


@patch("catanatron.state.roll_dice")
def test_rolling_a_seven_triggers_default_discard_limit(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    until_nine = 9 - player_num_resource_cards(game.state, players[1].color)
    player_deck_replenish(game.state, players[1].color, WHEAT, until_nine)
    assert player_num_resource_cards(game.state, players[1].color) == 9
    game.play_tick()  # should be player 0 rolling.

    assert len(game.state.playable_actions) == 1
    assert game.state.playable_actions == [
        Action(players[1].color, ActionType.DISCARD, None)
    ]

    game.play_tick()
    assert player_num_resource_cards(game.state, players[1].color) == 5


@patch("catanatron.state.roll_dice")
def test_all_players_discard_as_needed(fake_roll_dice):
    """Tests irrespective of who rolls the 7, all players discard"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    ordered_players = game.state.players
    fake_roll_dice.return_value = (3, 3)
    game.play_tick()  # should be p0 rolling a 6
    game.play_tick()  # should be p0 ending turn

    # fill everyones hand
    until_nine = 9 - player_num_resource_cards(game.state, players[0].color)
    player_deck_replenish(game.state, players[0].color, WHEAT, until_nine)
    until_nine = 9 - player_num_resource_cards(game.state, players[1].color)
    player_deck_replenish(game.state, players[1].color, WHEAT, until_nine)
    until_nine = 9 - player_num_resource_cards(game.state, players[2].color)
    player_deck_replenish(game.state, players[2].color, WHEAT, until_nine)
    until_nine = 9 - player_num_resource_cards(game.state, players[3].color)
    player_deck_replenish(game.state, players[3].color, WHEAT, until_nine)
    fake_roll_dice.return_value = (1, 6)
    game.play_tick()  # should be p1 rolling a 7

    # the following assumes, no matter who rolled 7, asking players
    #   to discard, happens in original seating-order.
    assert len(game.state.playable_actions) == 1
    assert game.state.playable_actions == [
        Action(ordered_players[0].color, ActionType.DISCARD, None)
    ]

    game.play_tick()  # p0 discards, places p1 in line to discard
    assert player_num_resource_cards(game.state, ordered_players[0].color) == 5
    assert len(game.state.playable_actions) == 1
    assert game.state.playable_actions == [
        Action(ordered_players[1].color, ActionType.DISCARD, None)
    ]

    game.play_tick()
    assert player_num_resource_cards(game.state, ordered_players[1].color) == 5
    assert len(game.state.playable_actions) == 1
    assert game.state.playable_actions == [
        Action(ordered_players[2].color, ActionType.DISCARD, None)
    ]

    game.play_tick()
    assert player_num_resource_cards(game.state, ordered_players[2].color) == 5
    assert len(game.state.playable_actions) == 1
    assert game.state.playable_actions == [
        Action(ordered_players[3].color, ActionType.DISCARD, None)
    ]

    game.play_tick()  # p3 discards, game goes back to p1 moving robber
    assert player_num_resource_cards(game.state, ordered_players[3].color) == 5
    assert game.state.is_moving_knight
    assert all(a.color == ordered_players[1].color for a in game.state.playable_actions)
    assert all(
        a.action_type == ActionType.MOVE_ROBBER for a in game.state.playable_actions
    )


@patch("catanatron.state.roll_dice")
def test_discard_is_configurable(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, discard_limit=10)
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    until_nine = 9 - player_num_resource_cards(game.state, players[1].color)
    player_deck_replenish(game.state, players[1].color, WHEAT, until_nine)
    assert player_num_resource_cards(game.state, players[1].color) == 9
    game.play_tick()  # should be p0 rolling.

    assert game.state.playable_actions != [
        Action(players[1].color, ActionType.DISCARD, None)
    ]


@patch("catanatron.state.roll_dice")
def test_end_turn_goes_to_next_player(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)  # not a 7

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    actions = []
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        actions.append(game.play_tick())

    p0_color = game.state.colors[0]
    p1_color = game.state.colors[1]
    assert (
        game.state.current_prompt == ActionPrompt.PLAY_TURN
        and game.state.current_color() == p0_color
    )
    assert game.state.playable_actions == [Action(p0_color, ActionType.ROLL, None)]

    game.execute(Action(p0_color, ActionType.ROLL, None))
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.current_color() == p0_color
    assert player_has_rolled(game.state, p0_color)
    assert Action(p0_color, ActionType.END_TURN, None) in game.state.playable_actions

    game.execute(Action(p0_color, ActionType.END_TURN, None))
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.current_color() == p1_color
    assert not player_has_rolled(game.state, p0_color)
    assert not player_has_rolled(game.state, p1_color)
    assert game.state.playable_actions == [Action(p1_color, ActionType.ROLL, None)]


# ===== Development Cards
def test_play_year_of_plenty_not_enough_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    player_to_act = players[0]
    game = Game(players)
    game.state.resource_freqdeck = [0, 0, 0, 0, 0]
    player_deck_replenish(game.state, player_to_act.color, YEAR_OF_PLENTY)

    action_to_execute = Action(
        player_to_act.color,
        ActionType.PLAY_YEAR_OF_PLENTY,
        [ORE, WHEAT],
    )

    with pytest.raises(ValueError):  # not enough cards in bank
        game.execute(action_to_execute)


def test_play_year_of_plenty_no_year_of_plenty_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    action_to_execute = Action(
        players[0].color, ActionType.PLAY_YEAR_OF_PLENTY, [ORE, WHEAT]
    )

    with pytest.raises(ValueError):  # no year of plenty card
        game.execute(action_to_execute)


def test_play_monopoly_no_monopoly_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    action_to_execute = Action(players[0].color, ActionType.PLAY_MONOPOLY, ORE)

    with pytest.raises(ValueError):  # no monopoly
        game.execute(action_to_execute)


@patch("catanatron.state.roll_dice")
def test_play_road_building(fake_roll_dice):
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    p0 = game.state.players[0]
    player_deck_replenish(game.state, p0.color, ROAD_BUILDING)

    # play initial phase
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    # roll not a 7
    fake_roll_dice.return_value = (1, 2)
    game.play_tick()  # roll

    game.execute(Action(p0.color, ActionType.PLAY_ROAD_BUILDING, None))
    assert game.state.is_road_building
    assert game.state.free_roads_available == 2
    game.play_tick()
    assert game.state.is_road_building
    assert game.state.free_roads_available == 1
    game.play_tick()
    assert not game.state.is_road_building
    assert game.state.free_roads_available == 0


def test_longest_road_steal():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    p0, p1 = game.state.players
    p0_key = player_key(game.state, p0.color)
    p1_key = player_key(game.state, p1.color)
    board = game.state.board

    # p0 has a road of length 4
    board.build_settlement(p0.color, 6, True)
    board.build_road(p0.color, (6, 7))
    board.build_road(p0.color, (7, 8))
    board.build_road(p0.color, (8, 9))
    board.build_road(p0.color, (9, 10))
    game.state.player_state[f'{p0_key}_VICTORY_POINTS'] = 1
    game.state.player_state[f'{p0_key}_ACTUAL_VICTORY_POINTS'] = 1

    # p1 has longest road of lenght 5
    board.build_settlement(p1.color, 28, True)
    board.build_road(p1.color, (27, 28))
    board.build_road(p1.color, (28, 29))
    board.build_road(p1.color, (29, 30))
    board.build_road(p1.color, (30, 31))
    board.build_road(p1.color, (31, 32))
    game.state.player_state[f'{p1_key}_VICTORY_POINTS'] = 3
    game.state.player_state[f'{p1_key}_ACTUAL_VICTORY_POINTS'] = 3
    game.state.player_state[f'{p1_key}_HAS_ROAD'] = True

    # Required to be able to apply actions other than rolling or initial build phase.
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    game.state.is_initial_build_phase = False
    game.state.player_state[f'{p0_key}_HAS_ROLLED'] = True
    game.state.playable_actions = generate_playable_actions(game.state)

    # Set up player0 to build two roads and steal longest road.
    road1 = (10, 11)
    road2 = (11, 12)
    player_deck_replenish(game.state, p0.color, WOOD, 2)
    player_deck_replenish(game.state, p0.color, BRICK, 2)

    # Matching length of longest road does not steal longest road.
    apply_action(game.state, Action(p0.color, ActionType.BUILD_ROAD, road1))
    assert game.state.player_state[f'{p0_key}_LONGEST_ROAD_LENGTH'] == 5
    assert game.state.player_state[f'{p0_key}_HAS_ROAD'] == False
    assert game.state.player_state[f'{p0_key}_VICTORY_POINTS'] == 1
    assert game.state.player_state[f'{p0_key}_ACTUAL_VICTORY_POINTS'] == 1
    assert game.state.player_state[f'{p1_key}_LONGEST_ROAD_LENGTH'] == 5
    assert game.state.player_state[f'{p1_key}_HAS_ROAD'] == True
    assert game.state.player_state[f'{p1_key}_VICTORY_POINTS'] == 3
    assert game.state.player_state[f'{p1_key}_ACTUAL_VICTORY_POINTS'] == 3

    # Surpassing length of longest road steals longest road and VPs.
    apply_action(game.state, Action(p0.color, ActionType.BUILD_ROAD, road2))
    assert game.state.player_state[f'{p0_key}_LONGEST_ROAD_LENGTH'] == 6
    assert game.state.player_state[f'{p0_key}_HAS_ROAD'] == True
    assert game.state.player_state[f'{p0_key}_VICTORY_POINTS'] == 3
    assert game.state.player_state[f'{p0_key}_ACTUAL_VICTORY_POINTS'] == 3
    assert game.state.player_state[f'{p1_key}_LONGEST_ROAD_LENGTH'] == 5
    assert game.state.player_state[f'{p1_key}_HAS_ROAD'] == False
    assert game.state.player_state[f'{p1_key}_VICTORY_POINTS'] == 1
    assert game.state.player_state[f'{p1_key}_ACTUAL_VICTORY_POINTS'] == 1


def test_second_placement_takes_cards_from_bank():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    assert sum(game.state.resource_freqdeck) == 19 * 5

    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    assert sum(game.state.resource_freqdeck) < 19 * 5


def test_vps_to_win_config():
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
    ]
    game = Game(players, vps_to_win=4)
    game.play()

    winning_color = game.winning_color()
    vps = get_actual_victory_points(game.state, winning_color)
    assert vps >= 4 and vps < 6


def test_cant_trade_same_resources_or_give():
    offering = [1, 0, 0, 0, 0]
    asking = [1, 0, 0, 0, 0]
    action_value = tuple([*offering, *asking])
    assert not is_valid_trade(action_value)

    offering = [0, 1, 0, 0, 0]
    asking = [0, 2, 0, 0, 0]
    action_value = tuple([*offering, *asking])
    assert not is_valid_trade(action_value)

    offering = [0, 1, 3, 0, 0]
    asking = [0, 0, 1, 0, 0]
    action_value = tuple([*offering, *asking])
    assert not is_valid_trade(action_value)


def test_cant_give_away_resources():
    offering = [1, 0, 0, 0, 0]
    asking = [0, 0, 0, 0, 0]
    action_value = tuple([*offering, *asking])
    assert not is_valid_trade(action_value)

    offering = [0, 0, 0, 0, 0]
    asking = [0, 2, 0, 0, 1]
    action_value = tuple([*offering, *asking])
    assert not is_valid_trade(action_value)


def test_trade_offers_are_valid():
    offering = [1, 0, 0, 0, 0]
    asking = [0, 1, 0, 0, 0]
    action_value = tuple([*offering, *asking])
    assert is_valid_trade(action_value)

    offering = [0, 0, 1, 0, 0]
    asking = [0, 2, 0, 0, 1]
    action_value = tuple([*offering, *asking])
    assert is_valid_trade(action_value)

    offering = [0, 0, 0, 2, 0]
    asking = [0, 1, 0, 0, 0]
    action_value = tuple([*offering, *asking])
    assert is_valid_trade(action_value)

    offering = [0, 0, 1, 1, 0]
    asking = [0, 1, 0, 0, 0]
    action_value = tuple([*offering, *asking])
    assert is_valid_trade(action_value)


@patch("catanatron.state.roll_dice")
def test_trading_sequence(fake_roll_dice):
    # Play initial building phase
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players)
    [p0, p1, p2] = game.state.players
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    # create 1:1 trade
    freqdeck = get_player_freqdeck(game.state, p0.color)
    index_of_a_resource_owned = next(i for i, v in enumerate(freqdeck) if v > 0)
    missing_resource_index = freqdeck.index(
        0
    )  # assumes its impossible to have one of each resource in first turn
    offered = [0, 0, 0, 0, 0]
    offered[index_of_a_resource_owned] = 1
    asking = [0, 0, 0, 0, 0]
    asking[missing_resource_index] = 1
    trade_action_value = tuple([*offered, *asking])
    action = Action(p0.color, ActionType.OFFER_TRADE, trade_action_value)

    # apply action, and listen to p1.decide_trade
    with pytest.raises(ValueError):  # can't offer trades before rolling. must risk 7
        game.execute(action)

    # roll not a 7
    fake_roll_dice.return_value = (1, 2)
    game.play_tick()
    freqdeck = get_player_freqdeck(game.state, p0.color)

    # test 1: players deny trade
    p1.decide = MagicMock(
        return_value=Action(p1.color, ActionType.REJECT_TRADE, (*trade_action_value, 0))
    )
    p2.decide = MagicMock(
        return_value=Action(p2.color, ActionType.REJECT_TRADE, (*trade_action_value, 0))
    )
    game.execute(action)  # now you can offer trades
    assert game.state.is_resolving_trade
    assert all(a.color == p1.color for a in game.state.playable_actions)
    assert all(
        a.action_type in [ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE]
        for a in game.state.playable_actions
    )

    # assert they asked players to accept/deny trade
    game.play_tick()  # ask p1 to decide
    game.play_tick()  # ask p2 to decide
    p1.decide.assert_called_once()
    p2.decide.assert_called_once()
    # assert trade didn't happen and is back at PLAY_TURN
    assert freqdeck == get_player_freqdeck(game.state, p0.color)
    assert not game.state.is_resolving_trade
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN

    # test 2: one of them (p1) accepts trade, but p0 regrets
    # ensure p1 has cards
    player_deck_replenish(game.state, p1.color, RESOURCES[missing_resource_index], 1)
    p1.decide = MagicMock(
        return_value=Action(p1.color, ActionType.ACCEPT_TRADE, (*trade_action_value, 0))
    )
    p2.decide = MagicMock(
        return_value=Action(p2.color, ActionType.REJECT_TRADE, (*trade_action_value, 0))
    )
    p0.decide = MagicMock(return_value=Action(p0.color, ActionType.CANCEL_TRADE, None))
    game.execute(action)
    assert game.state.is_resolving_trade
    game.play_tick()  # ask p1 to accept/reject
    game.play_tick()  # ask p2 to accept/reject
    game.play_tick()  # ask p1 to confirm
    p1.decide.assert_called_once()
    p2.decide.assert_called_once()
    p0.decide.assert_called_once_with(
        game,
        [
            Action(p0.color, ActionType.CANCEL_TRADE, None),
            Action(p0.color, ActionType.CONFIRM_TRADE, (*trade_action_value, p1.color)),
        ],
    )
    # assert trade didn't happen
    assert freqdeck == get_player_freqdeck(game.state, p0.color)
    assert not game.state.is_resolving_trade
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN

    # test 3: both of them accepts trade, p0 selects p2
    # ensure p1 and p2 have cards
    player_deck_replenish(game.state, p1.color, RESOURCES[missing_resource_index], 1)
    player_deck_replenish(game.state, p2.color, RESOURCES[missing_resource_index], 1)
    p1.decide = MagicMock(
        return_value=Action(p1.color, ActionType.ACCEPT_TRADE, (*trade_action_value, 0))
    )
    p2.decide = MagicMock(
        return_value=Action(p2.color, ActionType.ACCEPT_TRADE, (*trade_action_value, 0))
    )
    p0.decide = MagicMock(
        return_value=Action(
            p0.color, ActionType.CONFIRM_TRADE, (*trade_action_value, p2.color)
        )
    )
    game.execute(action)
    assert game.state.is_resolving_trade
    game.play_tick()  # ask p1 to accept/reject
    game.play_tick()  # ask p2 to accept/reject
    game.play_tick()  # ask p1 to confirm
    p1.decide.assert_called_once()
    p2.decide.assert_called_once()
    p0.decide.assert_called_once_with(
        game,
        [
            Action(p0.color, ActionType.CANCEL_TRADE, None),
            Action(p0.color, ActionType.CONFIRM_TRADE, (*trade_action_value, p1.color)),
            Action(p0.color, ActionType.CONFIRM_TRADE, (*trade_action_value, p2.color)),
        ],
    )
    # assert trade did happen
    expected = freqdeck[:]
    expected[index_of_a_resource_owned] -= 1
    expected[missing_resource_index] += 1
    assert get_player_freqdeck(game.state, p0.color) == expected
