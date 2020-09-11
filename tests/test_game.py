from catanatron.game import Game, playable_actions
from catanatron.models.board import Board
from catanatron.models.board_initializer import NodeRef, EdgeRef
from catanatron.models.board_algorithms import longest_road, continuous_roads_by_player
from catanatron.models.enums import ActionType, Action
from catanatron.models.player import Player, Color, SimplePlayer


def test_initial_build_phase():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.play_initial_build_phase()

    # assert there are 4 houses and 4 roads
    assert len(set(game.board.buildings.keys())) == (len(players) * 4)
    for path in continuous_roads_by_player(game.board, players[0]):
        assert len(path) == 1


def test_playable_actions():
    board = Board()
    player = Player(Color.RED)

    actions = playable_actions(player, False, board)
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.ROLL


def test_play_tick():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.play_initial_build_phase()
    game.play_tick()
    game.play_tick()

    # assert no exception thrown


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
