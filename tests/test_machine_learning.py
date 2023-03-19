import math
import random

import numpy as np

from tests.utils import advance_to_play_turn, build_initial_placements
from catanatron.state import player_deck_replenish
from catanatron.models.enums import ORE, Action, ActionType, WHEAT
from catanatron.models.board import Board, get_edges
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_EDGES,
    NUM_NODES,
    CatanMap,
    NodeRef,
)
from catanatron.game import Game
from catanatron.models.map import number_probability
from catanatron.models.player import SimplePlayer, Color
from catanatron_gym.features import (
    create_sample,
    expansion_features,
    port_features,
    reachability_features,
    iter_players,
    port_distance_features,
    resource_hand_features,
    tile_features,
    graph_features,
)
from catanatron_gym.board_tensor_features import (
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
    color = game.state.colors[0]
    game.execute(Action(color, ActionType.BUILD_SETTLEMENT, 3))
    game.execute(Action(color, ActionType.BUILD_ROAD, (2, 3)))

    ports = game.state.board.map.port_nodes
    se_port_resource = next(filter(lambda entry: 29 in entry[1], ports.items()))[0]
    port_name = "3:1" if se_port_resource is None else se_port_resource

    features = port_distance_features(game, color)
    assert features["P0_HAS_WHEAT_PORT"] == False
    assert features[f"P0_{port_name}_PORT_DISTANCE"] == 3


def test_resource_hand_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
    ]
    game = Game(players)

    red_index = game.state.color_to_index[Color.RED]
    game.state.player_state[f"P{red_index}_WHEAT_IN_HAND"] = 20
    player_deck_replenish(game.state, Color.BLUE, "ORE", 17)

    features = resource_hand_features(game, Color.RED)
    assert features["P0_WHEAT_IN_HAND"] == 20
    assert features["P1_NUM_RESOURCES_IN_HAND"] == 17

    features = resource_hand_features(game, Color.BLUE)
    assert features["P0_ORE_IN_HAND"] == 17
    assert features["P1_NUM_RESOURCES_IN_HAND"] == 20


def test_expansion_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    color = game.state.colors[0]
    game.execute(Action(color, ActionType.BUILD_SETTLEMENT, 3))
    game.execute(Action(color, ActionType.BUILD_ROAD, (2, 3)))

    neighbor_tile_resource = game.state.board.map.land_tiles[(1, -1, 0)].resource
    if neighbor_tile_resource is None:
        neighbor_tile_resource = game.state.board.map.land_tiles[(0, -1, 1)].resource

    features = expansion_features(game, color)
    assert features["P0_WHEAT_AT_DISTANCE_0"] == 0
    assert features[f"P0_{neighbor_tile_resource}_AT_DISTANCE_0"] == 0
    assert features[f"P0_{neighbor_tile_resource}_AT_DISTANCE_1"] > 0


def test_reachability_features():
    """Board in tensor-board-test.png"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    # NOTE: tensor-board-test.png is the board that happens after seeding random
    #   with 123 and running a random.sample() like so:
    # We do this here to allow Game.__init__ evolve freely.
    random.seed(123)
    random.sample(players, len(players))
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = Game(players, seed=123, catan_map=catan_map)
    p0_color = game.state.colors[0]

    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, 5))
    features = reachability_features(game, p0_color)
    assert features["P0_0_ROAD_REACHABLE_WOOD"] == number_probability(3)
    assert features["P0_0_ROAD_REACHABLE_BRICK"] == number_probability(4)
    assert features["P0_0_ROAD_REACHABLE_SHEEP"] == number_probability(6)
    assert features["P0_0_ROAD_REACHABLE_WHEAT"] == 0
    # these are 0 since cant build at distance 1
    assert features["P0_1_ROAD_REACHABLE_ORE"] == 0
    assert features["P0_1_ROAD_REACHABLE_WHEAT"] == 0
    # whats available at distance 0 should also be available at distance 1
    assert features["P0_1_ROAD_REACHABLE_WOOD"] == number_probability(3)
    assert features["P0_1_ROAD_REACHABLE_BRICK"] == number_probability(4)
    assert features["P0_1_ROAD_REACHABLE_SHEEP"] == number_probability(6)

    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (0, 5)))
    features = reachability_features(game, p0_color)
    assert features["P0_0_ROAD_REACHABLE_WOOD"] == number_probability(3)
    assert features["P0_0_ROAD_REACHABLE_BRICK"] == number_probability(4)
    assert features["P0_0_ROAD_REACHABLE_SHEEP"] == number_probability(6)
    assert features["P0_0_ROAD_REACHABLE_WHEAT"] == 0
    assert features["P0_1_ROAD_REACHABLE_ORE"] == number_probability(
        8
    ) + number_probability(5)
    assert features["P0_1_ROAD_REACHABLE_WHEAT"] == 2 * number_probability(9)

    # Test distance 2
    assert math.isclose(
        features["P0_2_ROAD_REACHABLE_ORE"],
        2 * number_probability(10)
        + 3 * number_probability(8)
        + 3 * number_probability(5),
    )

    # Test enemy making building removes buildability
    p1_color = game.state.colors[1]
    game.execute(Action(p1_color, ActionType.BUILD_SETTLEMENT, 1))
    features = reachability_features(game, p0_color)
    assert features["P0_1_ROAD_REACHABLE_ORE"] == number_probability(8)
    assert math.isclose(
        features["P0_2_ROAD_REACHABLE_ORE"],
        2 * number_probability(10) + 3 * number_probability(8),
    )


def test_tile_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    features = tile_features(game, players[0].color)
    tile = game.state.board.map.land_tiles[(0, 0, 0)]
    resource = tile.resource
    value = resource if resource is not None else "DESERT"
    proba = number_probability(tile.number) if resource is not None else 0
    assert features[f"TILE0_IS_{value}"]
    assert features[f"TILE0_PROBA"] == proba


def test_tile_features_in_mini():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
    ]
    game = Game(players, catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE))

    features = tile_features(game, players[0].color)
    haystack = "".join(features.keys())
    assert "TILE7" not in haystack


def test_port_features_in_mini():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
    ]
    game = Game(players, catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE))

    features = port_features(game, players[0].color)
    assert len(features) == 0


def test_graph_features():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    p0_color = game.state.colors[0]
    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, 3))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (2, 3)))

    features = graph_features(game, p0_color)
    assert features[f"NODE3_P0_SETTLEMENT"]
    assert features[f"EDGE(2, 3)_P0_ROAD"]
    assert not features[f"NODE3_P1_SETTLEMENT"]
    assert not features[f"NODE0_P1_SETTLEMENT"]
    assert len(features) == 54 * len(players) * 2 + NUM_EDGES * len(players)
    assert sum(features.values()) == 2

    haystack = "".join(features.keys())
    for edge in get_edges():
        assert str(edge) in haystack
    for node in range(NUM_NODES):
        assert ("NODE" + str(node)) in haystack


def test_graph_features_in_mini():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
    ]
    game = Game(players, catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE))
    p0_color = game.state.colors[0]
    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, 3))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (2, 3)))

    features = graph_features(game, p0_color)
    assert features[f"NODE3_P0_SETTLEMENT"]
    assert features[f"EDGE(2, 3)_P0_ROAD"]
    assert not features[f"NODE3_P1_SETTLEMENT"]
    assert not features[f"NODE0_P1_SETTLEMENT"]
    # todo: CHANGE NUM_EDGES
    assert len(features) == 24 * len(players) * 2 + 30 * len(players)
    assert sum(features.values()) == 2

    haystack = "".join(features.keys())
    for edge in get_edges(game.state.board.map.land_nodes):
        assert str(edge) in haystack
    for node in range(24):
        assert ("NODE" + str(node)) in haystack


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

    for coordinate in Board().map.land_tiles.keys():
        assert coordinate in tile_map


def test_create_board_tensor_channels_first():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    p0_color = game.state.colors[0]

    tensor = create_board_tensor(game, p0_color, True)
    assert tensor.shape == (20 - 4, 21, 11)


def test_create_board_tensor():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    p0_color = game.state.colors[0]

    # assert starts with no settlement/cities
    tensor = create_board_tensor(game, p0_color)
    assert tensor.shape == (21, 11, 20 - 4)
    assert tensor[0][0][0] == 0
    assert tensor[10][6][0] == 0
    assert tensor[9][6][0] == 0

    # assert settlement and road mark 1s correspondingly
    build_initial_placements(game, p0_actions=[3, (3, 4), 37, (14, 37)])
    tensor = create_board_tensor(game, p0_color)
    assert tensor[10][6][0] == 1
    assert tensor[9][6][1] == 1

    player_deck_replenish(game.state, p0_color, WHEAT, 2)
    player_deck_replenish(game.state, p0_color, ORE, 3)
    advance_to_play_turn(game)
    game.execute(Action(p0_color, ActionType.BUILD_CITY, 3))
    tensor = create_board_tensor(game, p0_color)
    assert tensor[10][6][0] == 2
    assert tensor[9][6][1] == 1


def test_robber_plane_simple():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)

    robber_channel = 13
    tensor = create_board_tensor(game, players[0].color)

    assert np.sum(tensor[:, :, robber_channel]) == 6
    assert np.max(tensor[:, :, robber_channel]) == 1


def test_resource_proba_planes():
    """Board in tensor-board-test.png"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    # NOTE: tensor-board-test.png is the board that happens after seeding random
    #   with 123 and running a random.sample() like so:
    # We do this here to allow Game.__init__ evolve freely.
    random.seed(123)
    random.sample(players, len(players))
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = Game(players, seed=123, catan_map=catan_map)

    tensor = create_board_tensor(game, players[0].color)
    assert tensor[0][0][0] == 0

    # Top left should be 0 for all resources. (water tile)
    for resource_channel in range(4, 9):
        top_left_tile = tensor[0:4, 0:2, resource_channel]
        assert np.all(np.equal(top_left_tile, 0))

    # Assert ten sheep left edge looks good
    sheep_channel = 10
    ten_sheep_left_edge = tensor[4, 0:3, sheep_channel]
    ten_proba = number_probability(10)
    assert np.sum(ten_sheep_left_edge) == ten_proba * 2  # 2 nodes

    # assert 5 wood top node has sheep too.
    wood_channel = 8
    five_proba = number_probability(5)
    five_wood_top_node = tensor[4, 2]
    assert np.all(np.equal(five_wood_top_node[sheep_channel], ten_proba))
    assert np.all(np.equal(five_wood_top_node[wood_channel], five_proba))

    # assert wood node adds up
    total_proba = five_proba + number_probability(11) + number_probability(3)
    middle_wood_node = tensor[4, 4]
    assert np.all(np.equal(middle_wood_node[wood_channel], total_proba))

    # assert brick tile has 6 non-zero node as expected
    four_proba = number_probability(4)
    assert tensor[6, 2, 9] == four_proba
    assert tensor[7, 2, 9] == 0.0
    assert tensor[8, 2, 9] == four_proba
    assert tensor[9, 2, 9] == 0.0
    assert tensor[10, 2, 9] == four_proba
    for i in range(5):
        assert tensor[6 + i, 3, 9] == 0.0
    assert tensor[6, 4, 9] == four_proba
    assert tensor[7, 4, 9] == 0.0
    assert tensor[8, 4, 9] == four_proba
    assert tensor[9, 4, 9] == 0.0
    assert tensor[10, 4, 9] == four_proba


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
    assert np.sum(tensor[:, :, -6:]) == 2 * 9

    # assert that 3:1 ports there are 4 * 2 nodes on.
    assert np.sum(tensor[:, :, -1]) == 2 * 4


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
    robber_tile = game.state.board.map.tiles[game.state.board.robber_coordinate]
    nw_desert_node = robber_tile.nodes[NodeRef.NORTHWEST]
    i, j = node_map[nw_desert_node]

    robber_plane_channel = 13
    expected = [
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    ]
    assert (
        np.transpose(tensor[i : i + 5, j : j + 3, robber_plane_channel]) == expected
    ).all()


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
        j, c = iter_players(tuple(game.state.colors), game.state.colors[i])[0]
        assert c == game.state.colors[i]

    # Test a specific case (p0=game.state.colors[0])
    iterator = iter_players(tuple(game.state.colors), game.state.colors[0])
    i, c = iterator[0]
    assert i == 0
    assert c == game.state.colors[0]
    i, c = iterator[1]
    assert i == 1
    assert c == game.state.colors[1]
    i, c = iterator[2]
    assert i == 2
    assert c == game.state.colors[2]
    i, c = iterator[3]
    assert i == 3
    assert c == game.state.colors[3]
