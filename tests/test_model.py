from catanatron.models import get_nodes_and_edges, Node, Edge, Tile, Resource


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

