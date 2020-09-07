from catanatron.models.board import Board
from catanatron.models.board_initializer import EdgeRef, NodeRef
from catanatron.models.board_algorithms import longest_road
from catanatron.models.player import Player, Color

# ===== Buildable nodes
def test_buildable_nodes():
    board = Board()
    nodes = board.buildable_nodes(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_nodes(Color.RED, initial_build_phase=True)
    assert len(nodes) == 54


def test_placing_settlement_removes_four_buildable_nodes():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    nodes = board.buildable_nodes(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_nodes(Color.RED, initial_build_phase=True)
    assert len(nodes) == 50
    nodes = board.buildable_nodes(Color.BLUE, initial_build_phase=True)
    assert len(nodes) == 50


def test_buildable_nodes_respects_distance_two():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )

    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHWEST)])
    nodes = board.buildable_nodes(Color.RED)
    assert len(nodes) == 0

    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.WEST)])
    nodes = board.buildable_nodes(Color.RED)
    assert len(nodes) == 1
    # TODO: assert Node?


# ===== Buildable edges
def test_buildable_edges_simple():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 3


def test_buildable_edges():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHWEST)])
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 4


# ===== Find connected components
def test_connected_components_empty_board():
    board = Board()
    components = board.find_connected_components(Color.RED)
    assert len(components) == 0


def test_one_connected_component():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHEAST)])
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.NORTHEAST)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.EAST)])
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.NORTHEAST)])
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1


def test_two_connected_components():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHWEST)])
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.NORTHEAST)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.NORTHEAST)])
    components = board.find_connected_components(Color.RED)
    assert len(components) == 2


def test_three_connected_components_bc_enemy_cut_road():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHWEST)])

    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.NORTHEAST)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.NORTHEAST)])
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.NORTHWEST)])
    board.build_road(Color.RED, board.edges[((-1, 1, 0), EdgeRef.NORTHEAST)])

    board.build_settlement(
        Color.BLUE,
        board.nodes[((0, 0, 0), NodeRef.NORTHWEST)],
        initial_build_phase=True,
    )
    components = board.find_connected_components(Color.RED)
    assert len(components) == 3
