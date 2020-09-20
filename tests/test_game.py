import pytest
from unittest.mock import MagicMock, patch

from catanatron.game import Game, yield_resources
from catanatron.models.board import Board
from catanatron.models.board_initializer import NodeRef, EdgeRef
from catanatron.models.board_algorithms import longest_road, continuous_roads_by_player
from catanatron.models.enums import Resource
from catanatron.models.actions import ActionType, Action
from catanatron.models.player import Player, Color, SimplePlayer
from catanatron.models.decks import ResourceDeck


def test_initial_build_phase():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.play_initial_build_phase()

    # assert there are 4 houses and 4 roads
    assert len(set(game.board.buildings.keys())) == (len(players) * 4)

    # assert should be house-road pairs, or together
    paths = continuous_roads_by_player(game.board, players[0])
    assert len(paths) == 1 or (
        len(paths) == 2 and len(paths[0]) == 1 and len(paths[1]) == 1
    )

    # assert should have resources from last house.
    assert players[0].resource_deck.num_cards() >= 1
    assert players[0].resource_deck.num_cards() <= 3
    assert players[1].resource_deck.num_cards() >= 1
    assert players[1].resource_deck.num_cards() <= 3


def test_can_play_for_a_bit():  # assert no exception thrown
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.play_initial_build_phase()

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
    game.play_initial_build_phase()

    players[1].resource_deck = ResourceDeck(empty=True)
    players[1].resource_deck.replenish(7, Resource.WHEAT)
    game.execute(Action(players[0], ActionType.ROLL, None))  # roll
    assert len(game.tick_queue) == 0


@patch("catanatron.game.roll_dice")
def test_rolling_a_seven_triggers_discard_mechanism(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.play_initial_build_phase()

    players[1].resource_deck = ResourceDeck(empty=True)
    players[1].resource_deck.replenish(9, Resource.WHEAT)
    game.execute(Action(players[0], ActionType.ROLL, None))  # roll
    assert len(game.tick_queue) == 1
    game.play_tick()
    assert players[1].resource_deck.num_cards() == 5


# ===== Yield Resources
def test_yield_resources():
    board = Board()
    resource_deck = ResourceDeck()

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
    resource_deck = ResourceDeck()

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
    resource_deck = ResourceDeck()

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
    resource_deck = ResourceDeck()

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
    red.resource_deck += ResourceDeck()  # whole bank in hand
    blue.resource_deck += ResourceDeck()  # whole bank in hand

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red,
            ActionType.BUILD_SETTLEMENT,
            nodes[((0, 0, 0), NodeRef.SOUTH)],
        ),
        initial_build_phase=True,
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHEAST)])
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color is None

    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.EAST)]))
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHEAST)])
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHWEST)])
    )
    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.WEST)]))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 5


def test_longest_road_tie():
    red = Player(Color.RED)
    blue = Player(Color.BLUE)
    red.resource_deck += ResourceDeck()  # whole bank in hand
    blue.resource_deck += ResourceDeck()  # whole bank in hand

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red,
            ActionType.BUILD_SETTLEMENT,
            nodes[((0, 0, 0), NodeRef.SOUTH)],
        ),
        initial_build_phase=True,
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHEAST)])
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_SETTLEMENT,
            nodes[((0, 2, -2), NodeRef.SOUTH)],
        ),
        initial_build_phase=True,
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.SOUTHEAST)],
        )
    )

    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.EAST)]))
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHEAST)])
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHWEST)])
    )
    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.WEST)]))

    game.execute(Action(blue, ActionType.BUILD_ROAD, edges[((0, 2, -2), EdgeRef.EAST)]))
    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.NORTHEAST)],
        )
    )
    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.NORTHWEST)],
        )
    )
    game.execute(Action(blue, ActionType.BUILD_ROAD, edges[((0, 2, -2), EdgeRef.WEST)]))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED  # even if blue also has 5-road. red had it first
    assert len(path) == 5

    game.execute(
        Action(
            blue,
            ActionType.BUILD_ROAD,
            edges[((0, 2, -2), EdgeRef.SOUTHWEST)],
        )
    )
    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.BLUE
    assert len(path) == 6


# test: complicated circle around
def test_complicated_road():  # classic 8-like roads
    red = Player(Color.RED)
    blue = Player(Color.BLUE)
    red.resource_deck += ResourceDeck()  # whole bank in hand
    blue.resource_deck += ResourceDeck()  # whole bank in hand

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(red, ActionType.BUILD_SETTLEMENT, nodes[((0, 0, 0), NodeRef.SOUTH)]),
        initial_build_phase=True,
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHEAST)])
    )
    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.EAST)]))
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHEAST)])
    )
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.NORTHWEST)])
    )
    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.WEST)]))
    game.execute(
        Action(red, ActionType.BUILD_ROAD, edges[((0, 0, 0), EdgeRef.SOUTHWEST)])
    )

    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.SOUTHWEST)],
        )
    )
    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.SOUTHEAST)],
        )
    )
    game.execute(Action(red, ActionType.BUILD_ROAD, edges[((1, -1, 0), EdgeRef.EAST)]))
    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.NORTHEAST)],
        )
    )
    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((1, -1, 0), EdgeRef.NORTHWEST)],
        )
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 11

    game.execute(
        Action(
            red,
            ActionType.BUILD_ROAD,
            edges[((2, -2, 0), EdgeRef.SOUTHWEST)],
        )
    )

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 11
