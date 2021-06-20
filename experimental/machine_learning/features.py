from typing import Tuple
import functools
from collections import Counter

import networkx as nx

from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.board import STATIC_GRAPH, get_edges, get_node_distances
from catanatron.models.map import NUM_NODES, NUM_TILES
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Resource,
    BuildingType,
    ActionType,
    VICTORY_POINT,
)
from catanatron.game import Game
from catanatron.models.map import number_probability


# ===== Helpers
def is_building(game, node_id, color, building_type):
    building = game.state.board.buildings.get(node_id, None)
    if building is None:
        return False
    else:
        return building[0] == color and building[1] == building_type


def is_road(game, edge, color):
    return game.state.board.get_edge_color(edge) == color


@functools.lru_cache(1024)
def iter_players(colors: Tuple[Color], p0_color: Color):
    """Iterator: for i, player in iter_players(game, p0.color)"""
    start_index = colors.index(p0_color)
    result = []
    for i in range(len(colors)):
        actual_index = (start_index + i) % len(colors)
        result.append((i, colors[actual_index]))
    return result


# ===== Extractors
def player_features(game, p0_color):
    # P0_ACTUAL_VPS
    # P{i}_PUBLIC_VPS, P1_PUBLIC_VPS, ...
    # P{i}_HAS_ARMY, P{i}_HAS_ROAD, P1_HAS_ARMY, ...
    # P{i}_ROADS_LEFT, P{i}_SETTLEMENTS_LEFT, P{i}_CITIES_LEFT, P1_...
    # P{i}_HAS_ROLLED, P{i}_LONGEST_ROAD_LENGTH
    features = dict()
    for i, color in iter_players(game.state.colors, p0_color):
        key = player_key(game.state, color)
        if color == p0_color:
            features["P0_ACTUAL_VPS"] = game.state.player_state[
                key + "_ACTUAL_VICTORY_POINTS"
            ]

        features[f"P{i}_PUBLIC_VPS"] = game.state.player_state[key + "_VICTORY_POINTS"]
        features[f"P{i}_HAS_ARMY"] = game.state.player_state[key + "_HAS_ARMY"]
        features[f"P{i}_HAS_ROAD"] = game.state.player_state[key + "_HAS_ROAD"]
        features[f"P{i}_ROADS_LEFT"] = game.state.player_state[key + "_ROADS_AVAILABLE"]
        features[f"P{i}_SETTLEMENTS_LEFT"] = game.state.player_state[
            key + "_SETTLEMENTS_AVAILABLE"
        ]
        features[f"P{i}_CITIES_LEFT"] = game.state.player_state[
            key + "_CITIES_AVAILABLE"
        ]
        features[f"P{i}_HAS_ROLLED"] = game.state.player_state[key + "_HAS_ROLLED"]
        features[f"P{i}_LONGEST_ROAD_LENGTH"] = game.state.player_state[
            key + "_LONGEST_ROAD_LENGTH"
        ]

    return features


def resource_hand_features(game, p0_color):
    # P0_WHEATS_IN_HAND, P0_WOODS_IN_HAND, ...
    # P0_ROAD_BUILDINGS_IN_HAND, P0_KNIGHT_IN_HAND, ..., P0_VPS_IN_HAND
    # P0_ROAD_BUILDINGS_PLAYABLE, P0_KNIGHT_PLAYABLE, ...
    # P0_ROAD_BUILDINGS_PLAYED, P0_KNIGHT_PLAYED, ...

    # P1_ROAD_BUILDINGS_PLAYED, P1_KNIGHT_PLAYED, ...
    # TODO: P1_WHEATS_INFERENCE, P1_WOODS_INFERENCE, ...
    # TODO: P1_ROAD_BUILDINGS_INFERENCE, P1_KNIGHT_INFERENCE, ...

    state = game.state
    player_state = state.player_state

    features = {}
    for i, color in iter_players(game.state.colors, p0_color):
        key = player_key(game.state, color)

        if color == p0_color:
            for resource in RESOURCES:
                features[f"P0_{resource}_IN_HAND"] = player_state[
                    key + f"_{resource}_IN_HAND"
                ]
            for card in DEVELOPMENT_CARDS:
                features[f"P0_{card}_IN_HAND"] = player_state[key + f"_{card}_IN_HAND"]
            features[f"P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = player_state[
                key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
            ]

        for card in DEVELOPMENT_CARDS:
            if card == VICTORY_POINT:
                continue  # cant play VPs
            features[f"P{i}_{card}_PLAYED"] = player_state[key + f"_PLAYED_{card}"]

        features[f"P{i}_NUM_RESOURCES_IN_HAND"] = player_num_resource_cards(
            state, color
        )
        features[f"P{i}_NUM_DEVS_IN_HAND"] = player_num_dev_cards(state, color)

    return features


@functools.lru_cache(NUM_TILES)  # one for each robber
def map_tile_features(catan_map, robber_coordinate):
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    features = {}

    for tile_id in range(NUM_TILES):
        tile = catan_map.tiles_by_id[tile_id]
        for resource in Resource:
            features[f"TILE{tile_id}_IS_{resource.value}"] = tile.resource == resource
        features[f"TILE{tile_id}_IS_DESERT"] = tile.resource == None
        features[f"TILE{tile_id}_PROBA"] = (
            0 if tile.resource is None else number_probability(tile.number)
        )
        features[f"TILE{tile_id}_HAS_ROBBER"] = (
            catan_map.tiles[robber_coordinate] == tile
        )
    return features


def tile_features(game, p0_color):
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    return map_tile_features(game.state.board.map, game.state.board.robber_coordinate)


@functools.lru_cache(1)
def map_port_features(catan_map):
    features = {}
    for port_id, port in catan_map.ports_by_id.items():
        for resource in Resource:
            features[f"PORT{port_id}_IS_{resource.value}"] = port.resource == resource
        features[f"PORT{port_id}_IS_THREE_TO_ONE"] = port.resource is None
    return features


def port_features(game, p0_color):
    # PORT0_WOOD, PORT0_THREE_TO_ONE, ...
    return map_port_features(game.state.board.map)


@functools.lru_cache(4)
def initialize_graph_features_template(num_players):
    features = {}
    for i in range(num_players):
        for node_id in range(NUM_NODES):
            for building in [BuildingType.SETTLEMENT, BuildingType.CITY]:
                features[f"NODE{node_id}_P{i}_{building.value}"] = False
        for edge in get_edges():
            features[f"EDGE{edge}_P{i}_ROAD"] = False
    return features


@functools.lru_cache(1024 * 2 * 2 * 2)
def get_node_hot_encoded(player_index, colors, settlements, cities, roads):
    features = {}

    for node_id in settlements:
        features[f"NODE{node_id}_P{player_index}_SETTLEMENT"] = True
    for node_id in cities:
        features[f"NODE{node_id}_P{player_index}_CITY"] = True
    for edge in roads:
        features[f"EDGE{tuple(sorted(edge))}_P{player_index}_ROAD"] = True

    return features


def graph_features(game, p0_color):
    features = initialize_graph_features_template(len(game.state.colors)).copy()

    for i, color in iter_players(game.state.colors, p0_color):
        settlements = tuple(
            game.state.buildings_by_color[color][BuildingType.SETTLEMENT]
        )
        cities = tuple(game.state.buildings_by_color[color][BuildingType.CITY])
        roads = tuple(game.state.buildings_by_color[color][BuildingType.ROAD])
        to_update = get_node_hot_encoded(
            i, game.state.colors, settlements, cities, roads
        )
        features.update(to_update)

    return features


def build_production_features(consider_robber):
    prefix = "EFFECTIVE_" if consider_robber else "TOTAL_"

    def production_features(game, p0_color):
        # P0_WHEAT_PRODUCTION, P0_ORE_PRODUCTION, ..., P1_WHEAT_PRODUCTION, ...
        features = {}
        board = game.state.board
        robbed_nodes = set(board.map.tiles[board.robber_coordinate].nodes.values())
        for resource in Resource:
            for i, color in iter_players(game.state.colors, p0_color):
                production = 0
                for node_id in get_player_buildings(
                    game.state, color, BuildingType.SETTLEMENT
                ):
                    if consider_robber and node_id in robbed_nodes:
                        continue
                    production += get_node_production(
                        game.state.board, node_id, resource
                    )
                for node_id in get_player_buildings(
                    game.state, color, BuildingType.CITY
                ):
                    if consider_robber and node_id in robbed_nodes:
                        continue
                    production += 2 * get_node_production(
                        game.state.board, node_id, resource
                    )
                features[f"{prefix}P{i}_{resource.value}_PRODUCTION"] = production

        return features

    return production_features


# @functools.lru_cache(maxsize=None)
def get_node_production(board, node_id, resource):
    tiles = board.map.adjacent_tiles[node_id]
    return sum([number_probability(t.number) for t in tiles if t.resource == resource])


def get_player_expandable_nodes(game, color):
    node_sets = game.state.board.find_connected_components(color)
    enemies = [enemy for enemy in game.state.players if enemy.color != color]
    enemy_node_ids = set()
    for enemy in enemies:
        enemy_node_ids.update(
            get_player_buildings(game.state, enemy.color, BuildingType.SETTLEMENT)
        )
        enemy_node_ids.update(
            get_player_buildings(game.state, enemy.color, BuildingType.CITY)
        )

    expandable_node_ids = [
        node_id
        for node_set in node_sets
        for node_id in node_set
        if node_id not in enemy_node_ids  # not plowed
    ]  # not exactly "buildable_node_ids" b.c. we could expand from non-buildable nodes
    return expandable_node_ids


REACHABLE_FEATURES_MAX = 3  # exclusive


def reachability_features(game, p0_color, levels=REACHABLE_FEATURES_MAX):
    features = {}

    board_buildable = game.state.board.buildable_node_ids(p0_color, True)
    for i, color in iter_players(game.state.colors, p0_color):
        owned_or_buildable = frozenset(
            get_player_buildings(game.state, color, BuildingType.SETTLEMENT)
            + get_player_buildings(game.state, color, BuildingType.CITY)
            + board_buildable
        )

        # Do layer 0
        zero_nodes = set()
        for component in game.state.board.connected_components[color]:
            for node_id in component:
                zero_nodes.add(node_id)

        production = count_production(
            frozenset(owned_or_buildable.intersection(zero_nodes)), game.state.board
        )
        for resource in Resource:
            features[f"P{i}_0_ROAD_REACHABLE_{resource.value}"] = production[resource]

        # for layers deep:
        last_layer_nodes = zero_nodes
        for level in range(1, levels):
            level_nodes = set(last_layer_nodes)
            for node_id in last_layer_nodes:
                if game.state.board.is_enemy_node(node_id, color):
                    continue  # not expandable.

                def can_follow_edge(neighbor_id):
                    edge = (node_id, neighbor_id)
                    return not game.state.board.is_enemy_road(edge, color)

                expandable = filter(can_follow_edge, STATIC_GRAPH.neighbors(node_id))
                level_nodes.update(expandable)

            production = count_production(
                frozenset(owned_or_buildable.intersection(level_nodes)),
                game.state.board,
            )
            for resource in Resource:
                features[f"P{i}_{level}_ROAD_REACHABLE_{resource.value}"] = production[
                    resource
                ]

            last_layer_nodes = level_nodes

    return features


@functools.lru_cache(maxsize=1000)
def count_production(nodes, board):
    production = Counter()
    for node_id in nodes:
        production += board.map.node_production[node_id]
    return production


def expansion_features(game, p0_color):
    global STATIC_GRAPH
    MAX_EXPANSION_DISTANCE = 3  # exclusive

    features = {}

    # For each connected component node, bfs_edges (skipping enemy edges and nodes nodes)
    empty_edges = set(get_edges())
    for i, color in iter_players(game.state.colors, p0_color):
        empty_edges.difference_update(
            get_player_buildings(game.state, color, BuildingType.ROAD)
        )
    searchable_subgraph = STATIC_GRAPH.edge_subgraph(empty_edges)

    board_buildable_node_ids = game.state.board.buildable_node_ids(
        p0_color, True
    )  # this should be the same for all players. TODO: Can maintain internally (instead of re-compute).

    for i, color in iter_players(game.state.colors, p0_color):
        expandable_node_ids = get_player_expandable_nodes(game, color)

        def skip_blocked_by_enemy(neighbor_ids):
            for node_id in neighbor_ids:
                node_color = game.state.board.get_node_color(node_id)
                if node_color is None or node_color == color:
                    yield node_id  # not owned by enemy, can explore

        # owned_edges = get_player_buildings(state, color, BuildingType.ROAD)
        dis_res_prod = {
            distance: {k: 0 for k in Resource}
            for distance in range(MAX_EXPANSION_DISTANCE)
        }
        for node_id in expandable_node_ids:
            if node_id in board_buildable_node_ids:  # node itself is buildable
                for resource in Resource:
                    production = get_node_production(
                        game.state.board, node_id, resource
                    )
                    dis_res_prod[0][resource] = max(
                        production, dis_res_prod[0][resource]
                    )

            if node_id not in searchable_subgraph.nodes():
                continue  # must be internal node, no need to explore

            bfs_iteration = nx.bfs_edges(
                searchable_subgraph,
                node_id,
                depth_limit=MAX_EXPANSION_DISTANCE - 1,
                sort_neighbors=skip_blocked_by_enemy,
            )

            paths = {node_id: []}
            for edge in bfs_iteration:
                a, b = edge
                path_until_now = paths[a]
                distance = len(path_until_now) + 1
                paths[b] = paths[a] + [b]

                if b not in board_buildable_node_ids:
                    continue

                # means we can get to node b, at distance=d, starting from path[0]
                for resource in Resource:
                    production = get_node_production(game.state.board, b, resource)
                    dis_res_prod[distance][resource] = max(
                        production, dis_res_prod[distance][resource]
                    )

        for distance, res_prod in dis_res_prod.items():
            for resource, prod in res_prod.items():
                features[f"P{i}_{resource.value}_AT_DISTANCE_{int(distance)}"] = prod

    return features


def port_distance_features(game, p0_color):
    # P0_HAS_WHEAT_PORT, P0_WHEAT_PORT_DISTANCE, ..., P1_HAS_WHEAT_PORT,
    features = {}
    ports = game.state.board.map.port_nodes
    distances = get_node_distances()
    for resource_or_none in list(Resource) + [None]:
        port_name = "3:1" if resource_or_none is None else resource_or_none.value
        for i, color in iter_players(game.state.colors, p0_color):
            expandable_node_ids = get_player_expandable_nodes(game, color)
            if len(expandable_node_ids) == 0:
                features[f"P{i}_HAS_{port_name}_PORT"] = False
                features[f"P{i}_{port_name}_PORT_DISTANCE"] = float("inf")
            else:
                min_distance = min(
                    [
                        distances[port_node_id][my_node]
                        for my_node in expandable_node_ids
                        for port_node_id in ports[resource_or_none]
                    ]
                )
                features[f"P{i}_HAS_{port_name}_PORT"] = min_distance == 0
                features[f"P{i}_{port_name}_PORT_DISTANCE"] = min_distance
    return features


def game_features(game, p0_color):
    # BANK_WOODS, BANK_WHEATS, ..., BANK_DEV_CARDS
    possibilities = set([a.action_type for a in game.state.playable_actions])
    features = {
        "BANK_DEV_CARDS": game.state.development_deck.num_cards(),
        "IS_MOVING_ROBBER": ActionType.MOVE_ROBBER in possibilities,
        "IS_DISCARDING": ActionType.DISCARD in possibilities,
    }
    for resource in Resource:
        features[f"BANK_{resource.value}"] = game.state.resource_deck.count(resource)
    return features


feature_extractors = [
    # PLAYER FEATURES =====
    player_features,
    resource_hand_features,
    # TRANSFERABLE BOARD FEATURES =====
    # build_production_features(True),
    # build_production_features(False),
    # expansion_features,
    # reachability_features,
    # RAW BASE-MAP FEATURES =====
    tile_features,
    port_features,
    graph_features,
    # GAME FEATURES =====
    game_features,
]


def create_sample(game, p0_color):
    record = {}
    for extractor in feature_extractors:
        record.update(extractor(game, p0_color))
    return record


def create_sample_vector(game, p0_color, features=None):
    features = features or get_feature_ordering()
    sample_dict = create_sample(game, p0_color)
    return [float(sample_dict[i]) for i in features]


FEATURE_ORDERING = None


def get_feature_ordering(num_players=4):
    global FEATURE_ORDERING
    if FEATURE_ORDERING is None:
        players = [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
        players = players[:num_players]
        game = Game(players)
        sample = create_sample(game, players[0].color)
        FEATURE_ORDERING = sorted(sample.keys())
    return FEATURE_ORDERING
