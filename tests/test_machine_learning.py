import tensorflow as tf

from catanatron.models.enums import Resource
from catanatron.models.board import Board, get_edges
from catanatron.models.map import NUM_NODES, NodeRef
from catanatron.models.actions import Action, ActionType
from catanatron.game import Game, number_probability
from catanatron.models.player import SimplePlayer, Color
from experimental.machine_learning.features import (
    create_sample,
    expansion_features,
    iter_players,
    port_distance_features,
    tile_features,
    graph_features,
)
from experimental.machine_learning.board_tensor_features import (
    create_board_tensor,
    get_node_and_edge_maps,
    init_board_tensor_map,
    init_tile_coordinate_map,
)


def test_create_sample():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    sample = create_sample(game, players[1].color)
    assert isinstance(sample, dict)
    assert len(sample) > 0


def test_port_distance_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.execute(Action(players[0].color, ActionType.BUILD_FIRST_SETTLEMENT, 3))
    game.execute(Action(players[0].color, ActionType.BUILD_INITIAL_ROAD, (3, 2)))

    ports = game.board.map.get_port_nodes()
    se_port_resource = next(filter(lambda entry: 29 in entry[1], ports.items()))[0]
    port_name = "3:1" if se_port_resource is None else se_port_resource.value

    features = port_distance_features(game, players[0].color)
    assert features["P0_HAS_WHEAT_PORT"] == False
    assert features[f"P0_{port_name}_PORT_DISTANCE"] == 3


def test_expansion_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.execute(Action(players[0].color, ActionType.BUILD_FIRST_SETTLEMENT, 3))
    game.execute(Action(players[0].color, ActionType.BUILD_INITIAL_ROAD, (3, 2)))

    neighbor_tile_resource = game.board.map.tiles[(1, -1, 0)].resource
    if neighbor_tile_resource is None:
        neighbor_tile_resource = game.board.map.tiles[(0, -1, 1)].resource

    features = expansion_features(game, players[0].color)
    assert features["P0_WHEAT_AT_DISTANCE_0"] == 0
    assert features[f"P0_{neighbor_tile_resource.value}_AT_DISTANCE_0"] == 0
    assert features[f"P0_{neighbor_tile_resource.value}_AT_DISTANCE_1"] > 0


def test_tile_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    features = tile_features(game, players[0].color)
    tile = game.board.map.tiles[(0, 0, 0)]
    resource = tile.resource
    value = resource.value if resource is not None else "DESERT"
    proba = number_probability(tile.number) if resource is not None else 0
    assert features[f"TILE0_IS_{value}"]
    assert features[f"TILE0_PROBA"] == proba


def test_graph_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.execute(Action(players[0].color, ActionType.BUILD_FIRST_SETTLEMENT, 3))
    game.execute(Action(players[0].color, ActionType.BUILD_INITIAL_ROAD, (3, 2)))

    features = graph_features(game, players[0].color)
    assert features[f"NODE3_P0_SETTLEMENT"]
    assert features[f"EDGE(2, 3)_P0_ROAD"]
    assert not features[f"NODE3_P1_SETTLEMENT"]
    assert not features[f"NODE0_P1_SETTLEMENT"]


def test_init_board_tensor_map():
    node_map, edge_map = init_board_tensor_map()
    assert node_map[82] == (0, 0)
    assert node_map[81] == (2, 0)
    assert node_map[93] == (20, 0)
    assert node_map[79] == (0, 2)
    assert node_map[43] == (4, 2)
    assert node_map[72] == (0, 10)
    assert node_map[60] == (20, 10)

    assert edge_map[(82, 81)] == (1, 0)
    assert edge_map[(81, 82)] == (1, 0)
    assert edge_map[(81, 47)] == (3, 0)
    assert edge_map[(92, 93)] == (19, 0)
    assert edge_map[(82, 79)] == (0, 1)
    assert edge_map[(47, 43)] == (4, 1)
    assert edge_map[(53, 94)] == (19, 2)
    assert edge_map[(44, 40)] == (2, 3)
    assert edge_map[(21, 16)] == (6, 3)
    assert edge_map[(24, 53)] == (18, 3)
    assert edge_map[(72, 71)] == (1, 10)
    assert edge_map[(60, 61)] == (19, 10)

    for i in range(NUM_NODES):
        assert i in node_map
    for edge in get_edges():
        assert edge in edge_map


def test_init_tile_map():
    tile_map = init_tile_coordinate_map()
    assert tile_map[(-1, 3, -2)] == (0, 0)
    assert tile_map[(0, 2, -2)] == (0, 4)
    assert tile_map[(-2, 2, 0)] == (4, 0)

    assert tile_map[(-1, 2, -1)] == (2, 2)  # first odd row

    assert tile_map[(0, 0, 0)] == (4, 8)  # center tile

    assert tile_map[(0, -2, 2)] == (8, 12)  # southeast

    for (coordinate, _) in Board().map.resource_tiles():
        assert coordinate in tile_map


def test_create_board_tensor():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    p0 = game.state.players[0]

    # assert starts with no settlement/cities
    tensor = create_board_tensor(game, p0.color)
    assert tensor.shape == (21, 11, 20)
    assert tensor[0][0][0] == 0
    assert tensor[10][6][0] == 0
    assert tensor[9][6][0] == 0

    # assert settlement marks a 1 in the spot
    game.execute(Action(p0.color, ActionType.BUILD_FIRST_SETTLEMENT, 3))
    tensor = create_board_tensor(game, p0.color)
    assert tensor.shape == (21, 11, 20)
    assert tensor[10][6][0] == 1
    assert tensor[9][6][0] == 0

    game.execute(Action(p0.color, ActionType.BUILD_INITIAL_ROAD, (3, 4)))
    tensor = create_board_tensor(game, p0.color)
    assert tensor[10][6][0] == 1
    assert tensor[9][6][1] == 1

    p0.resource_deck.replenish(2, Resource.WHEAT)
    p0.resource_deck.replenish(3, Resource.ORE)
    game.execute(Action(p0.color, ActionType.BUILD_CITY, 3))
    tensor = create_board_tensor(game, p0.color)
    assert tensor[10][6][0] == 2
    assert tensor[9][6][1] == 1


def test_robber_plane():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    robber_channel = 13
    tensor = create_board_tensor(game, players[0].color)

    assert tf.math.reduce_sum(tensor[:, :, robber_channel]) == 5 * 3
    assert tf.math.reduce_max(tensor[:, :, robber_channel]) == 1


def test_resource_proba_planes():
    """Board in tensor-board-test.png"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players, seed=123)
    tensor = create_board_tensor(game, players[0].color)
    assert tensor[0][0][0] == 0

    # Top left should be 0 for all resources. (water tile)
    for resource_channel in range(4, 9):
        top_left_tile = tensor[0:4, 0:2, resource_channel]
        assert tf.math.reduce_all(tf.math.equal(top_left_tile, 0)).numpy()

    # Assert ten sheep left edge looks good
    sheep_channel = 10
    ten_sheep_left_edge = tensor[4, 0:3, sheep_channel]
    ten_proba = number_probability(10)
    assert tf.math.reduce_sum(ten_sheep_left_edge) == ten_proba * 2  # 2 nodes

    # assert 5 wood top node has sheep too.
    wood_channel = 8
    five_proba = number_probability(5)
    five_wood_top_node = tensor[4, 2]
    assert tf.math.reduce_all(
        tf.math.equal(five_wood_top_node[sheep_channel], ten_proba)
    ).numpy()
    assert tf.math.reduce_all(
        tf.math.equal(five_wood_top_node[wood_channel], five_proba)
    ).numpy()

    # assert wood node adds up
    total_proba = five_proba + number_probability(11) + number_probability(3)
    middle_wood_node = tensor[4, 4]
    assert tf.math.reduce_all(
        tf.math.equal(middle_wood_node[wood_channel], total_proba)
    ).numpy()

    # assert brick tile has 6 non-zero node as expected
    four_proba = number_probability(4)
    tf.assert_equal(tensor[6, 2, 9], four_proba)
    tf.assert_equal(tensor[7, 2, 9], 0.0)
    tf.assert_equal(tensor[8, 2, 9], four_proba)
    tf.assert_equal(tensor[9, 2, 9], 0.0)
    tf.assert_equal(tensor[10, 2, 9], four_proba)
    for i in range(5):
        tf.assert_equal(tensor[6 + i, 3, 9], 0.0)
    tf.assert_equal(tensor[6, 4, 9], four_proba)
    tf.assert_equal(tensor[7, 4, 9], 0.0)
    tf.assert_equal(tensor[8, 4, 9], four_proba)
    tf.assert_equal(tensor[9, 4, 9], 0.0)
    tf.assert_equal(tensor[10, 4, 9], four_proba)


def test_port_planes():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    tensor = create_board_tensor(game, players[0].color)

    # assert there are 18 port nodes (4 3:1 and 5 resource)
    assert tf.math.reduce_sum(tensor[:, :, -6:]) == 2 * 9

    # assert that 3:1 ports there are 4 * 2 nodes on.
    assert tf.math.reduce_sum(tensor[:, :, -1]) == 2 * 4


def test_robber_plane():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    tensor = create_board_tensor(game, players[0].color)

    node_map, _ = get_node_and_edge_maps()
    robber_tile = game.board.map.tiles[game.board.robber_coordinate]
    nw_desert_node = robber_tile.nodes[NodeRef.NORTHWEST]
    i, j = node_map[nw_desert_node]

    robber_plane_channel = 13
    expected = [
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    ]
    tf.assert_equal(
        tf.transpose(tensor[i : i + 5, j : j + 3, robber_plane_channel]), expected
    )


def test_iter_players():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    # Test the firsts look good.
    for i in range(4):
        j, p = next(iter_players(game, game.state.players[i].color))
        assert p.color == game.state.players[i].color

    # Test a specific case (p0=game.state.players[0])
    iterator = iter_players(game, game.state.players[0].color)
    i, p = next(iterator)
    assert i == 0
    assert p.color == game.state.players[0].color
    i, p = next(iterator)
    assert i == 1
    assert p.color == game.state.players[1].color
    i, p = next(iterator)
    assert i == 2
    assert p.color == game.state.players[2].color
    i, p = next(iterator)
    assert i == 3
    assert p.color == game.state.players[3].color
