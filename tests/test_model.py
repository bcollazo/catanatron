import pytest

from catanatron.models import (
    get_nodes_and_edges,
    Node,
    Edge,
    Tile,
    Resource,
    Board,
    Color,
    EdgeRef,
    NodeRef,
)


def test_get_nodes_and_edges_on_empty_board():
    Node.next_autoinc_id = 0
    Edge.next_autoinc_id = 0
    nodes, edges = get_nodes_and_edges({}, (0, 0, 0))
    assert max(map(lambda n: n.id, nodes.values())) == 5
    assert max(map(lambda e: e.id, edges.values())) == 5


def test_get_nodes_and_edges_for_east_attachment():
    Node.next_autoinc_id = 0
    Edge.next_autoinc_id = 0
    nodes1, edges1 = get_nodes_and_edges({}, (0, 0, 0))
    nodes2, edges2 = get_nodes_and_edges(
        {(0, 0, 0): Tile(Resource.WOOD, 3, nodes1, edges1)}, (1, -1, 0)
    )
    assert max(map(lambda n: n.id, nodes2.values())) == 9
    assert max(map(lambda e: e.id, edges2.values())) == 10


def test_get_nodes_and_edges_for_east_and_southeast_attachment():
    Node.next_autoinc_id = 0
    Edge.next_autoinc_id = 0
    nodes1, edges1 = get_nodes_and_edges({}, (0, 0, 0))
    nodes2, edges2 = get_nodes_and_edges(
        {(0, 0, 0): Tile(Resource.WOOD, 3, nodes1, edges1)}, (1, -1, 0)
    )
    nodes3, edges3 = get_nodes_and_edges(
        {
            (0, 0, 0): Tile(Resource.WOOD, 3, nodes1, edges1),
            (1, -1, 0): Tile(Resource.BRICK, 6, nodes2, edges2),
        },
        (0, -1, 1),
    )
    assert max(map(lambda n: n.id, nodes3.values())) == 12
    assert max(map(lambda e: e.id, edges3.values())) == 14


def test_initial_placement_bypasses_restrictions():
    board = Board()
    with pytest.raises(ValueError):  # not connected and not initial-placement
        board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH)
    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHEAST)

    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)


def test_roads_must_always_be_connected():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)

    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (0, 0, 0), EdgeRef.EAST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHEAST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.EAST)
    board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHWEST)


def test_buildable_nodes():
    board = Board()
    nodes = board.buildable_nodes(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_nodes(Color.RED, initial_placement=True)
    assert len(nodes) == 54


def test_placing_settlement_removes_four_buildable_nodes():
    board = Board()
    board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
    nodes = board.buildable_nodes(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_nodes(Color.RED, initial_placement=True)
    assert len(nodes) == 50
    nodes = board.buildable_nodes(Color.BLUE, initial_placement=True)
    assert len(nodes) == 50


# def test_buildable_nodes_respects_distance_two():
#     board = Board()
#     board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)

#     board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHWEST)
#     nodes = board.buildable_nodes(Color.RED)
#     assert len(nodes) == 0

#     board.build_road(Color.RED, (0, 0, 0), EdgeRef.WEST)
#     nodes = board.buildable_nodes(Color.RED)
#     assert len(nodes) == 1
#     # TODO: assert Node?


# def test_must_build_distance_two():
#     board = Board()
#     board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
#     board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHEAST)

#     with pytest.raises(ValueError):  # distance less than 2
#         board.build_settlement(
#             Color.BLUE, (0, 0, 0), NodeRef.SOUTHWEST, initial_placement=True
#         )
#     board.build_settlement(
#         Color.BLUE, (0, 0, 0), NodeRef.NORTHEAST, initial_placement=True
#     )


# def test_placements_must_be_connected():
#     board = Board()
#     board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTH, initial_placement=True)
#     board.build_road(Color.RED, (0, 0, 0), EdgeRef.SOUTHEAST)

#     with pytest.raises(ValueError):  # distance less than 2 (even if connected)
#         board.build_settlement(Color.RED, (0, 0, 0), NodeRef.SOUTHEAST)
#     with pytest.raises(ValueError):  # not connected
#         board.build_settlement(Color.RED, (0, 0, 0), NodeRef.NORTHEAST)

#     board.build_road(Color.RED, (0, 0, 0), EdgeRef.EAST)
#     board.build_settlement(Color.RED, (0, 0, 0), NodeRef.NORTHEAST)
