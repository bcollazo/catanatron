import pytest

from catanatron.models.map import MINI_MAP_TEMPLATE, CatanMap
from catanatron.models.enums import RESOURCES
from catanatron.models.board import Board, get_node_distances
from catanatron.models.player import Color


def test_initial_build_phase_bypasses_restrictions():
    board = Board()
    with pytest.raises(ValueError):  # not connected and not initial-placement
        board.build_settlement(Color.RED, 3)
    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (3, 2))

    board.build_settlement(Color.RED, 3, initial_build_phase=True)


def test_roads_must_always_be_connected():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (3, 4))


def test_must_build_distance_two():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    with pytest.raises(ValueError):  # distance less than 2
        board.build_settlement(Color.BLUE, 4, initial_build_phase=True)
    board.build_settlement(Color.BLUE, 1, initial_build_phase=True)


def test_placements_must_be_connected():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    with pytest.raises(ValueError):  # distance less than 2 (even if connected)
        board.build_settlement(Color.RED, 2)
    with pytest.raises(ValueError):  # not connected
        board.build_settlement(Color.RED, 1)

    board.build_road(Color.RED, (2, 1))
    board.build_settlement(Color.RED, 1)


def test_city_requires_settlement_first():
    board = Board()
    with pytest.raises(ValueError):  # no settlement there
        board.build_city(Color.RED, 3)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_city(Color.RED, 3)


def test_calling_the_edge_differently_is_not_a_problem():
    """Tests building on (0,0,0), East is the same as (1,-1,0), West"""
    pass


def test_get_ports():
    board = Board()
    ports = board.map.port_nodes
    for resource in RESOURCES:
        assert len(ports[resource]) == 2
    assert len(ports[None]) == 8


def test_node_distances():
    node_distances = get_node_distances()
    assert node_distances[2][3] == 1

    # Test are symmetric
    assert node_distances[0][3] == 3
    assert node_distances[3][0] == 3

    assert node_distances[3][9] == 2
    assert node_distances[3][29] == 4

    assert node_distances[34][32] == 2
    assert node_distances[31][45] == 11


# ===== Buildable nodes
def test_buildable_nodes():
    board = Board()
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 54


def test_buildable_nodes_in_mini_map():
    board = Board(catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE, "random"))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 24


def test_placing_settlement_removes_four_buildable_nodes():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 50
    nodes = board.buildable_node_ids(Color.BLUE, initial_build_phase=True)
    assert len(nodes) == 50


def test_buildable_nodes_respects_distance_two():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    board.build_road(Color.RED, (3, 4))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0

    board.build_road(Color.RED, (4, 5))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 1
    assert nodes.pop() == 5


def test_cant_use_enemy_roads_to_connect():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    board.build_settlement(Color.BLUE, 1, initial_build_phase=True)
    board.build_road(Color.BLUE, (1, 2))
    board.build_road(Color.BLUE, (0, 1))
    board.build_road(Color.BLUE, (0, 20))  # north out of center tile

    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0

    nodes = board.buildable_node_ids(Color.BLUE)
    assert len(nodes) == 1


# ===== Buildable edges
def test_buildable_edges_simple():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 3


def test_buildable_edges_in_mini():
    board = Board(catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE, "random"))
    board.build_settlement(Color.RED, 19, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 2


def test_buildable_edges():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 4


def test_water_edge_is_not_buildable():
    board = Board()
    top_left_north_edge = 45
    board.build_settlement(Color.RED, top_left_north_edge, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 2


# ===== Find connected components
def test_connected_components_empty_board():
    board = Board()
    components = board.find_connected_components(Color.RED)
    assert len(components) == 0


def test_one_connected_component():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (1, 2))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_road(Color.RED, (0, 1))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1


def test_two_connected_components():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 2


def test_three_connected_components_bc_enemy_cut_road():
    board = Board()
    # Initial Building Phase of 2 players:
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))

    board.build_settlement(Color.BLUE, 15, initial_build_phase=True)
    board.build_road(Color.BLUE, (15, 4))
    board.build_settlement(Color.BLUE, 34, initial_build_phase=True)
    board.build_road(Color.BLUE, (34, 13))

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))

    # Extend road in a risky way
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (5, 16))

    # Plow </3
    board.build_road(Color.BLUE, (5, 4))
    board.build_settlement(Color.BLUE, 5)

    components = board.find_connected_components(Color.RED)
    assert len(components) == 3


def test_connected_components():
    board = Board()
    assert board.find_connected_components(Color.RED) == []

    # Simple test: roads stay at component, disconnected settlement creates new
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 2

    # This is just to be realistic
    board.build_settlement(Color.BLUE, 13, initial_build_phase=True)
    board.build_road(Color.BLUE, (13, 14))
    board.build_settlement(Color.BLUE, 37, initial_build_phase=True)
    board.build_road(Color.BLUE, (37, 14))

    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    assert len(board.find_connected_components(Color.RED)) == 2
    assert len(board.find_connected_components(Color.RED)[0]) == 2
    assert len(board.find_connected_components(Color.RED)[1]) == 2

    # Merging subcomponents
    board.build_road(Color.RED, (1, 2))
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 4

    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 15))
    board.build_road(Color.RED, (15, 17))
    assert len(board.find_connected_components(Color.RED)) == 1

    # Enemy cutoff
    board.build_road(Color.BLUE, (14, 15))
    board.build_settlement(Color.BLUE, 15)
    assert len(board.find_connected_components(Color.RED)) == 2


def test_building_road_to_enemy_works_well():
    board = Board()

    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (1, 0))

    # Test building towards enemy works well.
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 3


def test_building_into_enemy_doesnt_merge_components():
    board = Board()

    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 16, initial_build_phase=True)
    board.build_settlement(Color.RED, 6, initial_build_phase=True)
    board.build_road(Color.RED, (16, 5))
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (6, 1))
    board.build_road(Color.RED, (1, 0))
    assert len(board.find_connected_components(Color.RED)) == 2


def test_enemy_edge_not_buildable():
    board = Board()
    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_road(Color.BLUE, (0, 1))

    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, (2, 1))
    buildable_edges = board.buildable_edges(Color.RED)
    assert len(buildable_edges) == 3


def test_many_buildings():
    board = Board()
    board.build_settlement(Color.ORANGE, 7, True)
    board.build_settlement(Color.ORANGE, 12, True)
    board.build_road(Color.ORANGE, (6, 7))
    board.build_road(Color.ORANGE, (7, 8))
    board.build_road(Color.ORANGE, (8, 9))
    board.build_road(Color.ORANGE, (8, 27))
    board.build_road(Color.ORANGE, (26, 27))
    board.build_road(Color.ORANGE, (9, 10))
    board.build_road(Color.ORANGE, (10, 11))
    board.build_road(Color.ORANGE, (11, 12))
    board.build_road(Color.ORANGE, (12, 13))
    board.build_road(Color.ORANGE, (13, 34))
    assert len(board.find_connected_components(Color.ORANGE)) == 1

    board.build_settlement(Color.WHITE, 30, True)
    board.build_road(Color.WHITE, (29, 30))
    board.build_road(Color.WHITE, (10, 29))
    board.build_road(Color.WHITE, (28, 29))
    board.build_road(Color.WHITE, (27, 28))
    board.build_settlement(Color.WHITE, 10)  # cut
    board.build_road(Color.WHITE, (30, 31))
    board.build_road(Color.WHITE, (31, 32))
    board.build_settlement(Color.WHITE, 32)
    board.build_road(Color.WHITE, (11, 32))
    board.build_road(Color.WHITE, (32, 33))
    board.build_road(Color.WHITE, (33, 34))
    board.build_settlement(Color.WHITE, 34)
    board.build_road(Color.WHITE, (34, 35))
    board.build_road(Color.WHITE, (35, 36))

    board.build_settlement(Color.WHITE, 41, True)
    board.build_city(Color.WHITE, 41)
    board.build_road(Color.WHITE, (41, 42))
    board.build_road(Color.WHITE, (40, 42))
    board.build_settlement(Color.WHITE, 27)  # cut

    assert len(board.find_connected_components(Color.WHITE)) == 2
    assert len(board.find_connected_components(Color.ORANGE)) == 3


# TODO: Test super long road, cut at many places, to yield 5+ component graph


def test_cut_road_does_not_allow_building_out_of_enemy_settlement():
    board = Board()
    board.build_settlement(Color.BLUE, 46, initial_build_phase=True)
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_settlement(Color.ORANGE, 6, initial_build_phase=True)
    # RED's road ends at BLUE's settlement on 46.
    for edge in [(0, 20), (20, 22), (22, 49), (49, 48), (48, 46)]:
        board.build_road(Color.RED, edge)
    assert (19, 46) not in board.buildable_edges(Color.RED)

    # ORANGE cuts RED's road at 22.
    board.build_road(Color.ORANGE, (6, 23))
    board.build_road(Color.ORANGE, (23, 22))
    board.build_settlement(Color.ORANGE, 22)

    for component in board.find_connected_components(Color.RED):
        assert 22 not in component
        assert 46 not in component
    buildable = board.buildable_edges(Color.RED)
    assert (19, 46) not in buildable
    assert (45, 46) not in buildable
    with pytest.raises(ValueError):
        board.build_road(Color.RED, (46, 45))


def build_red_loop_around_tile(board):
    """RED settles at 4 and builds a loop 17-15-14-37-38-39 off it."""
    board.build_settlement(Color.RED, 4, initial_build_phase=True)
    board.build_road(Color.RED, (4, 15))
    for edge in [(17, 15), (15, 14), (14, 37), (37, 38), (38, 39), (39, 17)]:
        board.build_road(Color.RED, edge)


def test_cutting_a_loop_twice_keeps_enemy_settlements_out_of_components():
    board = Board()
    build_red_loop_around_tile(board)
    board.build_settlement(Color.BLUE, 35, initial_build_phase=True)
    board.build_road(Color.BLUE, (35, 36))
    board.build_road(Color.BLUE, (36, 37))
    board.build_settlement(Color.ORANGE, 42, initial_build_phase=True)
    board.build_road(Color.ORANGE, (42, 41))
    board.build_road(Color.ORANGE, (41, 39))

    board.build_settlement(Color.BLUE, 37)  # cutting the loop disconnects nothing
    board.build_settlement(Color.ORANGE, 39)  # leaves 38 on its own

    components = board.find_connected_components(Color.RED)
    assert sorted(sorted(c) for c in components) == [[4, 14, 15, 17], [38]]


def test_blocking_a_road_end_after_a_loop_cut_removes_it_from_every_component():
    board = Board()
    build_red_loop_around_tile(board)
    board.build_road(Color.RED, (14, 13))
    board.build_settlement(Color.BLUE, 35, initial_build_phase=True)
    board.build_road(Color.BLUE, (35, 36))
    board.build_road(Color.BLUE, (36, 37))
    board.build_settlement(Color.ORANGE, 11, initial_build_phase=True)
    board.build_road(Color.ORANGE, (11, 12))
    board.build_road(Color.ORANGE, (12, 13))

    board.build_settlement(Color.BLUE, 37)  # cutting the loop disconnects nothing
    board.build_settlement(Color.ORANGE, 13)  # on the end of RED's road

    for component in board.find_connected_components(Color.RED):
        assert 13 not in component
    assert (13, 34) not in board.buildable_edges(Color.RED)
