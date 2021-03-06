from catanatron.models.map import Tile, Resource, get_nodes_and_edges


def test_get_nodes_and_edges_on_empty_board():
    nodes, edges, node_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0)
    assert max(map(lambda n: n, nodes.values())) == 5


def test_get_nodes_and_edges_for_east_attachment():
    nodes1, edges1, node_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0)
    nodes2, edges2, node_autoinc = get_nodes_and_edges(
        {(0, 0, 0): Tile(0, Resource.WOOD, 3, nodes1, edges1)},
        (1, -1, 0),
        node_autoinc,
    )
    assert max(map(lambda n: n, nodes2.values())) == 9
    assert len(edges2.values()) == 6


def test_get_nodes_and_edges_for_east_and_southeast_attachment():
    nodes1, edges1, node_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0)
    nodes2, edges2, node_autoinc = get_nodes_and_edges(
        {(0, 0, 0): Tile(0, Resource.WOOD, 3, nodes1, edges1)},
        (1, -1, 0),
        node_autoinc,
    )
    nodes3, edges3, node_autoinc = get_nodes_and_edges(
        {
            (0, 0, 0): Tile(1, Resource.WOOD, 3, nodes1, edges1),
            (1, -1, 0): Tile(2, Resource.BRICK, 6, nodes2, edges2),
        },
        (0, -1, 1),
        node_autoinc,
    )
    assert max(map(lambda n: n, nodes3.values())) == 12
    assert len(edges3.values()) == 6
