import pytest

from catanatron.models.map import Tile, Resource
from catanatron.models.board_initializer import (
    get_nodes_and_edges,
    Node,
    Edge,
    EdgeRef,
    NodeRef,
)
from catanatron.models.board import Board
from catanatron.models.player import Color


def test_get_nodes_and_edges_on_empty_board():
    nodes, edges, node_autoinc, edge_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0, 0)
    assert max(map(lambda n: n.id, nodes.values())) == 5
    assert max(map(lambda e: e.id, edges.values())) == 5


def test_get_nodes_and_edges_for_east_attachment():
    nodes1, edges1, node_autoinc, edge_autoinc = get_nodes_and_edges(
        {}, (0, 0, 0), 0, 0
    )
    nodes2, edges2, node_autoinc, edge_autoinc = get_nodes_and_edges(
        {(0, 0, 0): Tile(0, Resource.WOOD, 3, nodes1, edges1)},
        (1, -1, 0),
        node_autoinc,
        edge_autoinc,
    )
    assert max(map(lambda n: n.id, nodes2.values())) == 9
    assert max(map(lambda e: e.id, edges2.values())) == 10


def test_get_nodes_and_edges_for_east_and_southeast_attachment():
    nodes1, edges1, node_autoinc, edge_autoinc = get_nodes_and_edges(
        {}, (0, 0, 0), 0, 0
    )
    nodes2, edges2, node_autoinc, edge_autoinc = get_nodes_and_edges(
        {(0, 0, 0): Tile(0, Resource.WOOD, 3, nodes1, edges1)},
        (1, -1, 0),
        node_autoinc,
        edge_autoinc,
    )
    nodes3, edges3, node_autoinc, edge_autoinc = get_nodes_and_edges(
        {
            (0, 0, 0): Tile(1, Resource.WOOD, 3, nodes1, edges1),
            (1, -1, 0): Tile(2, Resource.BRICK, 6, nodes2, edges2),
        },
        (0, -1, 1),
        node_autoinc,
        edge_autoinc,
    )
    assert max(map(lambda n: n.id, nodes3.values())) == 12
    assert max(map(lambda e: e.id, edges3.values())) == 14


def test_initial_build_phase_bypasses_restrictions():
    board = Board()
    with pytest.raises(ValueError):  # not connected and not initial-placement
        board.build_settlement(Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)])
    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHEAST)])

    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )


def test_roads_must_always_be_connected():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )

    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.EAST)])
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHEAST)])
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.EAST)])
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHWEST)])


def test_must_build_distance_two():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHEAST)])

    with pytest.raises(ValueError):  # distance less than 2
        board.build_settlement(
            Color.BLUE,
            board.nodes[((0, 0, 0), NodeRef.SOUTHWEST)],
            initial_build_phase=True,
        )
    board.build_settlement(
        Color.BLUE,
        board.nodes[((0, 0, 0), NodeRef.NORTHEAST)],
        initial_build_phase=True,
    )


def test_placements_must_be_connected():
    board = Board()
    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.SOUTHEAST)])

    with pytest.raises(ValueError):  # distance less than 2 (even if connected)
        board.build_settlement(Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTHEAST)])
    with pytest.raises(ValueError):  # not connected
        board.build_settlement(Color.RED, board.nodes[((0, 0, 0), NodeRef.NORTHEAST)])

    board.build_road(Color.RED, board.edges[((0, 0, 0), EdgeRef.EAST)])
    board.build_settlement(Color.RED, board.nodes[((0, 0, 0), NodeRef.NORTHEAST)])


def test_city_requires_settlement_first():
    board = Board()
    with pytest.raises(ValueError):  # no settlement there
        board.build_city(Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)])

    board.build_settlement(
        Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)], initial_build_phase=True
    )
    board.build_city(Color.RED, board.nodes[((0, 0, 0), NodeRef.SOUTH)])


def test_calling_the_edge_differently_is_not_a_problem():
    """Tests building on (0,0,0), East is the same as (1,-1,0), West"""
    pass


def test_get_ports():
    board = Board()
    ports = list(board.get_port_nodes())
    assert len(ports) == 9 * 2
