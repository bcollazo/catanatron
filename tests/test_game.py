from catanatron.game import (
    Game,
    playable_actions,
    yield_resources,
    city_possible_actions,
)
from catanatron.models.board import Board
from catanatron.models.board_initializer import NodeRef, EdgeRef
from catanatron.models.board_algorithms import longest_road, continuous_roads_by_player
from catanatron.models.enums import ActionType, Action
from catanatron.models.player import Player, Color, SimplePlayer
from catanatron.models.decks import ResourceDecks


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
    assert players[0].resource_decks.num_cards() >= 1
    assert players[0].resource_decks.num_cards() <= 3
    assert players[1].resource_decks.num_cards() >= 1
    assert players[1].resource_decks.num_cards() <= 3


def test_playable_actions():
    board = Board()
    player = Player(Color.RED)

    actions = playable_actions(player, False, board)
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.ROLL


def test_city_playable_actions():
    board = Board()
    player = Player(Color.RED)

    assert len(city_possible_actions(player, board)) == 0  # no money or place

    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    assert len(city_possible_actions(player, board)) == 0  # no money

    player.resource_decks += ResourceDecks.city_cost()
    assert len(city_possible_actions(player, board)) == 1

    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.NORTH)], initial_build_phase=True
    )
    assert len(city_possible_actions(player, board)) == 2


def test_can_play_for_a_bit():  # assert no exception thrown
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.play_initial_build_phase()

    for _ in range(10):
        game.play_tick()


# ===== Yield Resources
def test_yield_resources():
    board = Board()
    resource_decks = ResourceDecks()

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    payout, depleted = yield_resources(board, resource_decks, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 1


def test_yield_resources_two_settlements():
    board = Board()
    resource_decks = ResourceDecks()

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[(coordinate, EdgeRef.SOUTHWEST)])
    board.build_road(Color.RED, board.edges[(coordinate, EdgeRef.WEST)])
    board.build_settlement(Color.RED, board.nodes[(coordinate, NodeRef.NORTHWEST)])
    payout, depleted = yield_resources(board, resource_decks, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 2


def test_yield_resources_two_players_and_city():
    board = Board()
    resource_decks = ResourceDecks()

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
    payout, depleted = yield_resources(board, resource_decks, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 3
    assert payout[Color.BLUE].count(tile.resource) >= 2


def test_empty_payout_if_not_enough_resources():
    board = Board()
    resource_decks = ResourceDecks()

    tile, coordinate = board.tiles[(0, 0, 0)], (0, 0, 0)
    if tile.resource == None:  # is desert
        tile, coordinate = board.tiles[(1, -1, 0)], (1, -1, 0)

    board.build_settlement(
        Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_city(Color.RED, board.nodes[(coordinate, NodeRef.SOUTH)])
    resource_decks.draw(18, tile.resource)

    payout, depleted = yield_resources(board, resource_decks, tile.number)
    assert depleted == [tile.resource]
    assert Color.RED not in payout or payout[Color.RED].count(tile.resource) == 0


# ===== Longest road
def test_longest_road_simple():
    red = Player(Color.RED)
    blue = Player(Color.BLUE)

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
