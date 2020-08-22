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
    nodes, edges = get_nodes_and_edges({}, (0, 0, 0))
    assert max(map(lambda n: n.id, nodes.values())) == 5
    assert max(map(lambda e: e.id, edges.values())) == 5

    nodes, edges = get_nodes_and_edges(
        {(0, 0, 0): Tile(Resource.WOOD, 3, nodes, edges)}, (1, -1, 0)
    )
    assert max(map(lambda n: n.id, nodes.values())) == 9
    assert max(map(lambda e: e.id, edges.values())) == 10

