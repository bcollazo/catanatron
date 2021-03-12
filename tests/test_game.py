import pytest
from unittest.mock import MagicMock, patch

from catanatron.game import Game, yield_resources
from catanatron.algorithms import continuous_roads_by_player
from catanatron.models.board import Board
from catanatron.models.enums import Resource, DevelopmentCard, BuildingType
from catanatron.models.actions import ActionType, Action, ActionPrompt, TradeOffer
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.decks import ResourceDeck


def test_initial_build_phase():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    for i in range(len(game.state.tick_queue) - 1):
        game.play_tick()

    # assert there are 4 houses and 4 roads
    settlements = [
        i
        for building in game.state.board.buildings.values()
        if building[1] == BuildingType.SETTLEMENT
    ]
    assert len(settlements) == 4

    # assert should be house-road pairs, or together
    paths = continuous_roads_by_player(game.state.board, players[0].color)
    assert len(paths) == 1 or (
        len(paths) == 2 and len(paths[0]) == 1 and len(paths[1]) == 1
    )

    # assert should have resources from last house.
    # can only assert <= 3 b.c. player might place on a corner desert
    assert players[0].resource_deck.num_cards() <= 3
    assert players[1].resource_deck.num_cards() <= 3


def test_can_play_for_a_bit():  # assert no exception thrown
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    for _ in range(10):
        game.play_tick()


def test_buying_road_is_payed_for():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    game.state.board.build_road = MagicMock()
    action = Action(players[0].color, ActionType.BUILD_ROAD, (3, 4))
    with pytest.raises(ValueError):  # not enough money
        game.execute(action)

    players[0].resource_deck.replenish(1, Resource.WOOD)
    players[0].resource_deck.replenish(1, Resource.BRICK)
    game.execute(action)

    assert players[0].resource_deck.count(Resource.WOOD) == 0
    assert players[0].resource_deck.count(Resource.BRICK) == 0
    assert game.state.resource_deck.count(Resource.WOOD) == 20  # since we didnt yield


def test_moving_robber_steals_correctly():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    players[1].resource_deck.replenish(1, Resource.WHEAT)
    game.state.board.build_settlement(Color.BLUE, 3, initial_build_phase=True)

    action = Action(players[0].color, ActionType.MOVE_ROBBER, ((2, 0, -2), None, None))
    game.execute(action)
    assert players[0].resource_deck.num_cards() == 0
    assert players[1].resource_deck.num_cards() == 1

    action = Action(
        players[0].color,
        ActionType.MOVE_ROBBER,
        ((0, 0, 0), players[1].color, Resource.WHEAT),
    )
    game.execute(action)
    assert players[0].resource_deck.num_cards() == 1
    assert players[1].resource_deck.num_cards() == 0


@patch("catanatron.game.roll_dice")
def test_seven_cards_dont_trigger_discarding(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    blue_seating = game.state.players.index(players[1])

    players[1].resource_deck = ResourceDeck()
    players[1].resource_deck.replenish(7, Resource.WHEAT)
    game.execute(Action(players[0].color, ActionType.ROLL, None))  # roll

    discarding_ticks = list(
        filter(
            lambda a: a[0] == blue_seating and a[1] == ActionPrompt.DISCARD,
            game.state.tick_queue,
        )
    )
    assert len(discarding_ticks) == 0


@patch("catanatron.game.roll_dice")
def test_rolling_a_seven_triggers_discard_mechanism(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    blue_seating = game.state.players.index(players[1])
    for _ in range(8):
        game.play_tick()  # run initial placements

    players[1].resource_deck = ResourceDeck()
    players[1].resource_deck.replenish(9, Resource.WHEAT)
    game.play_tick()  # should be player 0 rolling.

    discarding_ticks = list(
        filter(
            lambda a: a[0] == blue_seating and a[1] == ActionPrompt.DISCARD,
            game.state.tick_queue,
        )
    )
    assert len(discarding_ticks) == 1

    game.play_tick()
    assert players[1].resource_deck.num_cards() == 5


# ===== Development Cards
def test_cant_buy_more_than_max_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    with pytest.raises(ValueError):  # not enough money
        game.execute(Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None))

    players[0].resource_deck.replenish(26, Resource.SHEEP)
    players[0].resource_deck.replenish(26, Resource.WHEAT)
    players[0].resource_deck.replenish(26, Resource.ORE)

    for i in range(25):
        game.execute(Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None))

    # assert must have all victory points
    game.count_victory_points()
    assert players[0].development_deck.num_cards() == 25
    assert players[0].public_victory_points == 0
    assert players[0].actual_victory_points == 5

    with pytest.raises(ValueError):  # not enough cards in bank
        game.execute(Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None))

    assert players[0].resource_deck.num_cards() == 3


def test_play_year_of_plenty_gives_player_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    player_to_act = players[0]
    player_to_act.development_deck.replenish(1, DevelopmentCard.YEAR_OF_PLENTY)

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act.color,
        ActionType.PLAY_YEAR_OF_PLENTY,
        [Resource.ORE, Resource.WHEAT],
    )

    game.execute(action_to_execute)

    for card_type in Resource:
        if card_type == Resource.ORE or card_type == Resource.WHEAT:
            assert player_to_act.resource_deck.count(card_type) == 1
            assert game.state.resource_deck.count(card_type) == 18
        else:
            assert player_to_act.resource_deck.count(card_type) == 0
            assert game.state.resource_deck.count(card_type) == 19
    assert player_to_act.development_deck.count(DevelopmentCard.YEAR_OF_PLENTY) == 0


def test_play_year_of_plenty_not_enough_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    player_to_act = players[0]
    game = Game(players)
    game.state.resource_deck = ResourceDeck()
    player_to_act.development_deck.replenish(1, DevelopmentCard.YEAR_OF_PLENTY)

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act.color,
        ActionType.PLAY_YEAR_OF_PLENTY,
        [Resource.ORE, Resource.WHEAT],
    )

    with pytest.raises(ValueError):  # not enough cards in bank
        game.execute(action_to_execute)


def test_play_year_of_plenty_no_year_of_plenty_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    players[0].clean_turn_state()
    action_to_execute = Action(
        players[0].color, ActionType.PLAY_YEAR_OF_PLENTY, [Resource.ORE, Resource.WHEAT]
    )

    with pytest.raises(ValueError):  # no year of plenty card
        game.execute(action_to_execute)


def test_play_monopoly_no_monopoly_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    players[0].clean_turn_state()
    action_to_execute = Action(players[0].color, ActionType.PLAY_MONOPOLY, Resource.ORE)

    with pytest.raises(ValueError):  # no monopoly
        game.execute(action_to_execute)


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
    game = Game(players)

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act.color, ActionType.PLAY_MONOPOLY, Resource.ORE
    )

    game.execute(action_to_execute)

    assert player_to_act.resource_deck.count(Resource.ORE) == 5
    assert player_to_steal_from_1.resource_deck.count(Resource.ORE) == 0
    assert player_to_steal_from_1.resource_deck.count(Resource.WHEAT) == 1
    assert player_to_steal_from_2.resource_deck.count(Resource.ORE) == 0
    assert player_to_steal_from_2.resource_deck.count(Resource.WHEAT) == 1


# ===== Yield Resources
def test_yield_resources():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile = board.map.tiles[(0, 0, 0)]
    if tile.resource is None:  # is desert
        tile = board.map.tiles[(-1, 0, 1)]

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    print(tile)
    print(payout, depleted)
    print(board.map.tiles)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 1


def test_yield_resources_two_settlements():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, edge2, node2 = board.map.tiles[(0, 0, 0)], (4, 5), 5
    if tile.resource is None:  # is desert
        tile, edge2, node2 = board.map.tiles[(-1, 0, 1)], (4, 15), 15

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, edge2)
    board.build_settlement(Color.RED, node2)
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 2


def test_yield_resources_two_players_and_city():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, edge1, edge2, red_node, blue_node = (
        board.map.tiles[(0, 0, 0)],
        (2, 3),
        (3, 4),
        4,
        0,
    )
    if tile.resource is None:  # is desert
        tile, edge1, edge2, red_node, blue_node = (
            board.map.tiles[(1, -1, 0)],
            (9, 2),
            (9, 8),
            8,
            6,
        )

    # red has one settlements and one city on tile
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, edge1)
    board.build_road(Color.RED, edge2)
    board.build_settlement(Color.RED, red_node)
    board.build_city(Color.RED, red_node)

    # blue has a city in tile
    board.build_settlement(Color.BLUE, blue_node, initial_build_phase=True)
    board.build_city(Color.BLUE, blue_node)
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 3
    assert payout[Color.BLUE].count(tile.resource) >= 2


def test_empty_payout_if_not_enough_resources():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile = board.map.tiles[(0, 0, 0)]
    if tile.resource is None:  # is desert
        tile = board.map.tiles[(-1, 0, 1)]

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_city(Color.RED, 3)
    resource_deck.draw(18, tile.resource)

    payout, depleted = yield_resources(board, resource_deck, tile.number)
    print(board.map.tiles)
    print(payout, depleted)
    print(resource_deck)
    assert depleted == [tile.resource]
    assert Color.RED not in payout or payout[Color.RED].count(tile.resource) == 0


def test_can_only_play_one_dev_card_per_turn():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    players[0].development_deck.replenish(2, DevelopmentCard.YEAR_OF_PLENTY)

    players[0].clean_turn_state()
    action = Action(
        players[0].color, ActionType.PLAY_YEAR_OF_PLENTY, 2 * [Resource.BRICK]
    )
    game.execute(action)
    with pytest.raises(ValueError):  # shouldnt be able to play two dev cards
        game.execute(action)


def test_trade_execution():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    players[0].resource_deck.replenish(4, Resource.BRICK)
    trade_offer = TradeOffer([Resource.BRICK] * 4, [Resource.ORE], None)
    action = Action(players[0].color, ActionType.MARITIME_TRADE, trade_offer)
    game.execute(action)

    assert players[0].resource_deck.num_cards() == 1
    assert game.state.resource_deck.num_cards() == 19 * 5 + 4 - 1


def test_can_trade_with_port():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    # Find port at (3, -3, 0), West.
    port_node_id = 25
    port = game.state.board.map.tiles[(3, -3, 0)]
    action = Action(players[0].color, ActionType.BUILD_FIRST_SETTLEMENT, port_node_id)
    game.execute(action)

    resource_out = port.resource or Resource.WHEAT
    num_out = 3 if port.resource is None else 2
    players[0].resource_deck.replenish(num_out, resource_out)
    resource_in = Resource.WHEAT if resource_out != Resource.WHEAT else Resource.WOOD

    actions = game.playable_actions(players[0], ActionPrompt.PLAY_TURN)
    trade_offer = TradeOffer([resource_out] * num_out, resource_in, None)
    assert len(actions) == 5
    # assert Action(players[0].color, ActionType.MARITIME_TRADE, trade_offer) in actions?


def test_second_placement_takes_cards_from_bank():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    assert len(game.state.resource_deck.to_array()) == 19 * 5

    action = Action(Color.RED, ActionType.BUILD_SECOND_SETTLEMENT, 0)
    game.execute(action)

    assert len(game.state.resource_deck.to_array()) < 19 * 5
