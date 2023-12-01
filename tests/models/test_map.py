from catanatron import WOOD, BRICK
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    CatanMap,
    LandTile,
    get_nodes_and_edges,
    get_node_counter_production,
    DICE_PROBAS,
)


def test_node_production_of_same_resource_adjacent_tile():
    # See https://github.com/bcollazo/catanatron/issues/263.
    adjacent_tiles = {
        1: [
            LandTile(1, WOOD, 8, dict(), dict()),
            LandTile(2, WOOD, 6, dict(), dict()),
            LandTile(3, WOOD, 12, dict(), dict()),
        ]
    }
    result = get_node_counter_production(adjacent_tiles, 1)
    assert result["WOOD"] == DICE_PROBAS[12] + DICE_PROBAS[6] + DICE_PROBAS[8]


def test_mini_map_can_be_created():
    mini = CatanMap.from_template(MINI_MAP_TEMPLATE)
    assert len(mini.land_tiles) == 7
    assert len(mini.land_nodes) == 24
    assert len(mini.tiles_by_id) == 7
    assert len(mini.ports_by_id) == 0
    assert len(mini.port_nodes) == 0
    assert len(mini.adjacent_tiles) == 24
    assert len(mini.node_production) == 24

    resources = [i.resource for i in mini.land_tiles.values()]
    assert any(isinstance(i, str) for i in resources)
    assert any(i is None for i in resources)  # theres one desert


def test_base_map_can_be_created():
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    assert len(catan_map.land_tiles) == 19
    assert len(catan_map.node_production) == 54


def test_get_nodes_and_edges_on_empty_board():
    nodes, edges, node_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0)
    assert max(map(lambda n: n, nodes.values())) == 5


def test_get_nodes_and_edges_for_east_attachment():
    nodes1, edges1, node_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0)
    nodes2, edges2, node_autoinc = get_nodes_and_edges(
        {(0, 0, 0): LandTile(0, WOOD, 3, nodes1, edges1)},
        (1, -1, 0),
        node_autoinc,
    )
    assert max(map(lambda n: n, nodes2.values())) == 9
    assert len(edges2.values()) == 6


def test_get_nodes_and_edges_for_east_and_southeast_attachment():
    nodes1, edges1, node_autoinc = get_nodes_and_edges({}, (0, 0, 0), 0)
    nodes2, edges2, node_autoinc = get_nodes_and_edges(
        {(0, 0, 0): LandTile(0, WOOD, 3, nodes1, edges1)},
        (1, -1, 0),
        node_autoinc,
    )
    nodes3, edges3, node_autoinc = get_nodes_and_edges(
        {
            (0, 0, 0): LandTile(1, WOOD, 3, nodes1, edges1),
            (1, -1, 0): LandTile(2, BRICK, 6, nodes2, edges2),
        },
        (0, -1, 1),
        node_autoinc,
    )
    assert max(map(lambda n: n, nodes3.values())) == 12
    assert len(edges3.values()) == 6
