import pytest
from unittest.mock import MagicMock, patch

from catanatron.game import Game, yield_resources, replay_game
from catanatron.algorithms import longest_road, continuous_roads_by_player
from catanatron.models.board import Board
from catanatron.models.board_initializer import NodeRef, EdgeRef
from catanatron.models.enums import Resource, DevelopmentCard
from catanatron.models.actions import ActionType, Action, ActionPrompt, TradeOffer
from catanatron.models.player import Player, Color, SimplePlayer
from catanatron.models.decks import ResourceDeck, DevelopmentDeck


def test_initial_build_phase():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    for i in range(len(game.tick_queue) - 1):
        game.play_tick()

    # assert there are 4 houses and 4 roads
    assert len(set(game.board.buildings.keys())) == (len(players) * 4)

    # assert should be house-road pairs, or together
    paths = continuous_roads_by_player(game.board, players[0])
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

    game.board.build_road = MagicMock()
    action = Action(
        players[0],
        ActionType.BUILD_ROAD,
        game.board.edges[((0, 0, 0), EdgeRef.SOUTHEAST)],
    )
    with pytest.raises(ValueError):  # not enough money
        game.execute(action)

    players[0].resource_deck.replenish(1, Resource.WOOD)
    players[0].resource_deck.replenish(1, Resource.BRICK)
    game.execute(action)

    assert players[0].resource_deck.count(Resource.WOOD) == 0
    assert players[0].resource_deck.count(Resource.BRICK) == 0
    assert game.resource_deck.count(Resource.WOOD) == 20  # since we didnt yield


def test_moving_robber_steals_correctly():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    players[1].resource_deck.replenish(1, Resource.WHEAT)
    game.board.build_settlement(
        Color.BLUE,
        game.board.nodes[((0, 0, 0), NodeRef.SOUTH)],
        initial_build_phase=True,
    )

    action = Action(players[0], ActionType.MOVE_ROBBER, ((2, 0, -2), None))
    game.execute(action)
    assert players[0].resource_deck.num_cards() == 0
    assert players[1].resource_deck.num_cards() == 1

    action = Action(players[0], ActionType.MOVE_ROBBER, ((0, 0, 0), players[1]))
    game.execute(action)
    assert players[0].resource_deck.num_cards() == 1
    assert players[1].resource_deck.num_cards() == 0


@patch("catanatron.game.roll_dice")
def test_seven_cards_dont_trigger_discarding(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    players[1].resource_deck = ResourceDeck()
    players[1].resource_deck.replenish(7, Resource.WHEAT)
    game.execute(Action(players[0], ActionType.ROLL, None))  # roll

    discarding_ticks = list(
        filter(
            lambda a: a[0] == players[1] and a[1] == ActionPrompt.DISCARD,
            game.tick_queue,
        )
    )
    assert len(discarding_ticks) == 0


@patch("catanatron.game.roll_dice")
def test_rolling_a_seven_triggers_discard_mechanism(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    for _ in range(8):
        game.play_tick()  # run initial placements

    players[1].resource_deck = ResourceDeck()
    players[1].resource_deck.replenish(9, Resource.WHEAT)
    game.play_tick()  # should be player 0 rolling.

    discarding_ticks = list(
        filter(
            lambda a: a[0] == players[1] and a[1] == ActionPrompt.DISCARD,
            game.tick_queue,
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
        game.execute(Action(players[0], ActionType.BUY_DEVELOPMENT_CARD, None))

    players[0].resource_deck.replenish(26, Resource.SHEEP)
    players[0].resource_deck.replenish(26, Resource.WHEAT)
    players[0].resource_deck.replenish(26, Resource.ORE)

    for i in range(25):
        game.execute(Action(players[0], ActionType.BUY_DEVELOPMENT_CARD, None))

    # assert must have all victory points
    game.count_victory_points()
    assert players[0].development_deck.num_cards() == 25
    assert players[0].public_victory_points == 0
    assert players[0].actual_victory_points == 5

    with pytest.raises(ValueError):  # not enough cards in bank
        game.execute(Action(players[0], ActionType.BUY_DEVELOPMENT_CARD, None))

    assert players[0].resource_deck.num_cards() == 3


def test_play_year_of_plenty_gives_player_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    player_to_act = players[0]
    player_to_act.development_deck.replenish(1, DevelopmentCard.YEAR_OF_PLENTY)
    cards_to_add = ResourceDeck()
    cards_to_add.replenish(1, Resource.ORE)
    cards_to_add.replenish(1, Resource.WHEAT)

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act, ActionType.PLAY_YEAR_OF_PLENTY, cards_to_add
    )

    game.execute(action_to_execute)

    for card_type in Resource:
        if card_type == Resource.ORE or card_type == Resource.WHEAT:
            assert player_to_act.resource_deck.count(card_type) == 1
            assert game.resource_deck.count(card_type) == 18
        else:
            assert player_to_act.resource_deck.count(card_type) == 0
            assert game.resource_deck.count(card_type) == 19
    assert player_to_act.development_deck.count(DevelopmentCard.YEAR_OF_PLENTY) == 0


def test_play_year_of_plenty_not_enough_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    player_to_act = players[0]
    game = Game(players)
    game.resource_deck = ResourceDeck()
    player_to_act.development_deck.replenish(1, DevelopmentCard.YEAR_OF_PLENTY)

    cards_to_add = ResourceDeck()
    cards_to_add.replenish(1, Resource.ORE)
    cards_to_add.replenish(1, Resource.WHEAT)

    player_to_act.clean_turn_state()
    action_to_execute = Action(
        player_to_act, ActionType.PLAY_YEAR_OF_PLENTY, cards_to_add
    )

    with pytest.raises(ValueError):  # not enough cards in bank
        game.execute(action_to_execute)


def test_play_year_of_plenty_no_year_of_plenty_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    cards_to_add = ResourceDeck()
    cards_to_add.replenish(1, Resource.ORE)
    cards_to_add.replenish(1, Resource.WHEAT)

    players[0].clean_turn_state()
    action_to_execute = Action(players[0], ActionType.PLAY_YEAR_OF_PLENTY, cards_to_add)

    with pytest.raises(ValueError):  # no year of plenty card
        game.execute(action_to_execute)


def test_play_monopoly_no_monopoly_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    players[0].clean_turn_state()
    action_to_execute = Action(players[0], ActionType.PLAY_MONOPOLY, Resource.ORE)

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
    action_to_execute = Action(player_to_act, ActionType.PLAY_MONOPOLY, Resource.ORE)

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

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 1


def test_yield_resources_two_settlements():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[(coordinate, EdgeRef.SOUTHWEST)])
    board.build_road(Color.RED, board.edges[(coordinate, EdgeRef.WEST)])
    board.build_settlement(Color.RED, board.nodes[(coordinate, NodeRef.NORTHWEST)])
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 2


def test_yield_resources_two_players_and_city():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    # red has one settlements and one city on tile
    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[(coordinate, EdgeRef.SOUTHWEST)])
    board.build_road(Color.RED, board.edges[(coordinate, EdgeRef.WEST)])
    board.build_settlement(Color.RED, board.nodes[(coordinate, NodeRef.NORTHWEST)])
    board.build_city(Color.RED, board.nodes[(coordinate, NodeRef.NORTHWEST)])

    # blue has a city in tile
    board.build_settlement(
        Color.BLUE,
        board.nodes[(coordinate, NodeRef.NORTHEAST)],
        initial_build_phase=True,
    )
    board.build_city(Color.BLUE, board.nodes[(coordinate, NodeRef.NORTHEAST)])
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 3
    assert payout[Color.BLUE].count(tile.resource) >= 2


def test_empty_payout_if_not_enough_resources():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_city(Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)])
    resource_deck.draw(18, tile.resource)

    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert depleted == [tile.resource]
    assert Color.RED not in payout or payout[Color.RED].count(tile.resource) == 0


# ===== Longest road
def test_longest_road_simple():
    red = Player(Color.RED)
    blue = Player(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red, ActionType.BUILD_FIRST_SETTLEMENT, nodes[((0, 0, 0), NodeRef.SOUTH)].id
        )
    )
    game.execute(
        Action(
            red, ActionType.BUILD_INITIAL_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id
        )
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color is None

    game.execute(
        Action(red, ActionType.BUILD_INITIAL_ROAD, edges[((0, 0, 0), EdgeRef.EAST)].id)
    )
    game.execute(
        Action(
            red, ActionType.BUILD_INITIAL_ROAD, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id
        )
    )
    game.execute(
        Action(
            red, ActionType.BUILD_INITIAL_ROAD, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id
        )
    )
    game.execute(
        Action(red, ActionType.BUILD_INITIAL_ROAD, edges[((0, 0, 0), EdgeRef.WEST)].id)
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 5


def test_longest_road_tie():
    red = Player(Color.RED)
    blue = Player(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red,
            ActionType.BUILD_FIRST_SETTLEMENT,
            nodes[((0, 0, 0), NodeRef.SOUTH)].id,
        ),
    )
    game.execute(
        Action(
            red, ActionType.BUILD_INITIAL_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id
        )
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_FIRST_SETTLEMENT,
            nodes[((0, 2, -2), NodeRef.SOUTH)].id,
        ),
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_INITIAL_ROAD,
            edges[((0, 2, -2), EdgeRef.SOUTHEAST)].id,
        )
    )

    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.EAST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.WEST)].id)
    )

    game.execute(
        Action(blue, ActionType.BUILD_ROAD, edges[((0, 2, -2), EdgeRef.EAST)].id)
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.NORTHEAST)].id,
        )
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.NORTHWEST)].id,
        )
    )
    game.execute(
        Action(blue, ActionType.BUILD_ROAD, edges[((0, 2, -2), EdgeRef.WEST)].id)
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED  # even if blue also has 5-road. red had it first
    assert len(path) == 5

    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.SOUTHWEST)].id,
        )
    )
    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.BLUE
    assert len(path) == 6


# test: complicated circle around
def test_complicated_road():  # classic 8-like roads
    red = Player(Color.RED)
    blue = Player(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red, ActionType.BUILD_FIRST_SETTLEMENT, nodes[((0, 0, 0), NodeRef.SOUTH)].id
        ),
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.EAST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.WEST)].id)
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHWEST)].id)
    )

    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.SOUTHWEST)].id,
        )
    )
    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.SOUTHEAST)].id,
        )
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((1, -1, 0), EdgeRef.EAST)].id)
    )
    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.NORTHEAST)].id,
        )
    )
    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.NORTHWEST)].id,
        )
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 11

    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((2, -2, 0), EdgeRef.SOUTHWEST)].id,
        )
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 11


def test_can_only_play_one_dev_card_per_turn():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    players[0].development_deck.replenish(2, DevelopmentCard.YEAR_OF_PLENTY)
    cards_selected = ResourceDeck()
    cards_selected.replenish(2, Resource.BRICK)

    players[0].clean_turn_state()
    action = Action(players[0], ActionType.PLAY_YEAR_OF_PLENTY, cards_selected)
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
    action = Action(players[0], ActionType.MARITIME_TRADE, trade_offer)
    game.execute(action)

    assert players[0].resource_deck.num_cards() == 1
    assert game.resource_deck.num_cards() == 19 * 5 + 4 - 1


def test_can_trade_with_port():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    # Find port at (3, -3, 0), West.
    port_node = game.board.nodes[((2, -2, 0), NodeRef.NORTHEAST)]
    port = game.board.tiles[(3, -3, 0)]
    action = Action(players[0], ActionType.BUILD_FIRST_SETTLEMENT, port_node.id)
    game.execute(action)

    resource_out = port.resource or Resource.WHEAT
    num_out = 3 if port.resource is None else 2
    players[0].resource_deck.replenish(num_out, resource_out)
    resource_in = Resource.WHEAT if resource_out != Resource.WHEAT else Resource.WOOD

    actions = game.playable_actions(players[0], ActionPrompt.PLAY_TURN)
    trade_offer = TradeOffer([resource_out] * num_out, resource_in, None)
    assert len(actions) == 5
    # assert Action(players[0], ActionType.MARITIME_TRADE, trade_offer) in actions?
