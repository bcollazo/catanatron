from catanatron.models.board import Board, Color, EdgeRef, NodeRef
from catanatron.models.board_algorithms import (
    buildable_nodes,
    buildable_edges,
    find_connected_components,
    longest_road,
)

# ===== Buildable nodes
def test_buildable_nodes():
    board = Board()
    nodes = buildable_nodes(board, Color.RED)
    assert len(nodes) == 0
    nodes = buildable_nodes(board, Color.RED, initial_placement=True)
    assert len(nodes) == 54


def test_placing_settlement_removes_four_buildable_nodes():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    nodes = buildable_nodes(board, Color.RED)
    assert len(nodes) == 0
    nodes = buildable_nodes(board, Color.RED, initial_placement=True)
    assert len(nodes) == 50
    nodes = buildable_nodes(board, Color.BLUE, initial_placement=True)
    assert len(nodes) == 50


def test_buildable_nodes_respects_distance_two():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)

    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHWEST)
    nodes = buildable_nodes(board, Color.RED)
    assert len(nodes) == 0

    board.build_road(Color.RED, (0, 0, 0), EdgeRef.WEST)
    nodes = buildable_nodes(board, Color.RED)
    assert len(nodes) == 1
    # TODO: assert Node?


# ===== Buildable edges
def test_buildable_edges_simple():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    buildable = buildable_edges(board, Color.RED)
    assert len(buildable) == 3


def test_buildable_edges():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHWEST)
    buildable = buildable_edges(board, Color.RED)
    assert len(buildable) == 4


# ===== Find connected components
def test_connected_components_empty_board():
    board = Board()
    components = find_connected_components(board, Color.RED)
    assert len(components) == 0


def test_one_connected_component():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHEAST)
    board.build_settlement(
        Color.RED, (0, 0, 0), NodeRef.NORTHEAST, initial_placement=True
    )
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.EAST)
    components = find_connected_components(board, Color.RED)
    assert len(components) == 1

    board.build_road(Color.RED, (0, 0, 0), EdgeRef.NORTHEAST)
    components = find_connected_components(board, Color.RED)
    assert len(components) == 1


def test_two_connected_components():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHWEST)
    components = find_connected_components(board, Color.RED)
    assert len(components) == 1

    board.build_settlement(
        Color.RED, (0, 0, 0), NodeRef.NORTHEAST, initial_placement=True
    )
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.NORTHEAST)
    components = find_connected_components(board, Color.RED)
    assert len(components) == 2


def test_three_connected_components_bc_enemy_cut_road():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHWEST)

    board.build_settlement(
        Color.RED, (0, 0, 0), NodeRef.NORTHEAST, initial_placement=True
    )
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.NORTHEAST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.NORTHWEST)
    board.build_road(Color.RED, (-1, 1, 0), EdgeRef.NORTHEAST)

    board.build_settlement(
        Color.BLUE, (0, 0, 0), NodeRef.NORTHWEST, initial_placement=True
    )
    components = find_connected_components(board, Color.RED)
    assert len(components) == 3


# ===== Longest road
def test_longest_road_simple():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHEAST)

    color = longest_road(board)
    assert color is None

    board.build_road(Color.RED, (0, 0, 0), EdgeRef.EAST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.NORTHEAST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.NORTHWEST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.WEST)

    color = longest_road(board)
    assert color == Color.RED


# test: some roads under 5, None
# test: single 5-road
# test: two-players with 5-roads
# test: circle around
# test: complicated circle around
