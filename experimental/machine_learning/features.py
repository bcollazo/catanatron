import functools
from typing import Generator, Tuple

import networkx as nx

from catanatron.models.board import get_edges, get_node_distances
from catanatron.models.map import NUM_NODES, NUM_TILES
from catanatron.models.player import Color, Player, SimplePlayer
from catanatron.models.enums import Resource, DevelopmentCard, BuildingType
from catanatron.game import Game, number_probability
from catanatron.algorithms import continuous_roads_by_player


# ===== Helpers
def port_is_resource(game, port_id, resource):
    port = game.board.get_port_by_id(port_id)
    return port.resource == resource


def port_is_threetoone(game, port_id):
    port = game.board.get_port_by_id(port_id)
    return port.resource is None


def is_building(game, node_id, player, building_type):
    node = game.board.nxgraph.nodes[node_id]
    building = node.get("building", None)
    if building is None:
        return False
    else:
        return node["color"] == player.color and building == building_type


def is_road(game, edge, player):
    return game.board.get_edge_color(edge) == player.color


def iter_players(
    game: Game, p0_color: Color
) -> Generator[Tuple[int, Player], any, any]:
    """Iterator: for i, player in iter_players(game, p0.color)"""
    p0_index = next(i for i, p in enumerate(game.state.players) if p.color == p0_color)
    for i in range(len(game.state.players)):
        player_index = (p0_index + i) % len(game.state.players)
        yield i, game.state.players[player_index]


# ===== Extractors
def player_features(game, p0_color):
    # P0_ACTUAL_VPS
    # P{i}_PUBLIC_VPS, P1_PUBLIC_VPS, ...
    # P{i}_HAS_ARMY, P{i}_HAS_ROAD, P1_HAS_ARMY, ...
    # P{i}_ROADS_LEFT, P{i}_SETTLEMENTS_LEFT, P{i}_CITIES_LEFT, P1_...
    # P{i}_HAS_ROLLED, P{i}_LONGEST_ROAD_LENGTH
    features = dict()
    for i, player in iter_players(game, p0_color):
        if player.color == p0_color:
            features["P0_ACTUAL_VPS"] = player.actual_victory_points

        features[f"P{i}_PUBLIC_VPS"] = player.public_victory_points
        features[f"P{i}_HAS_ARMY"] = player.has_army
        features[f"P{i}_HAS_ROAD"] = player.has_road
        features[f"P{i}_ROADS_LEFT"] = player.roads_available
        features[f"P{i}_SETTLEMENTS_LEFT"] = player.settlements_available
        features[f"P{i}_CITIES_LEFT"] = player.cities_available
        features[f"P{i}_HAS_ROLLED"] = player.has_rolled

        paths = continuous_roads_by_player(game.board, player.color)
        path_lengths = map(lambda path: len(path), paths)
        features[f"P{i}_LONGEST_ROAD_LENGTH"] = (
            0 if len(paths) == 0 else max(path_lengths)
        )

    return features


def resource_hand_features(game, p0_color):
    # P0_WHEATS_IN_HAND, P0_WOODS_IN_HAND, ...
    # P0_ROAD_BUILDINGS_IN_HAND, P0_KNIGHT_IN_HAND, ..., P0_VPS_IN_HAND
    # P0_ROAD_BUILDINGS_PLAYABLE, P0_KNIGHT_PLAYABLE, ...
    # P0_ROAD_BUILDINGS_PLAYED, P0_KNIGHT_PLAYED, ...

    # P1_ROAD_BUILDINGS_PLAYED, P1_KNIGHT_PLAYED, ...
    # TODO: P1_WHEATS_INFERENCE, P1_WOODS_INFERENCE, ...
    # TODO: P1_ROAD_BUILDINGS_INFERENCE, P1_KNIGHT_INFERENCE, ...

    features = {}
    for i, player in iter_players(game, p0_color):
        if player.color == p0_color:
            for resource in Resource:
                features[f"P0_{resource.value}_IN_HAND"] = player.resource_deck.count(
                    resource
                )
                for card in DevelopmentCard:
                    features[
                        f"P0_{card.value}_IN_HAND"
                    ] = player.development_deck.count(card)
                    if card != DevelopmentCard.VICTORY_POINT:
                        features[f"P0_{card.value}_PLAYABLE"] = (
                            card in player.playable_development_cards
                        )

        for card in DevelopmentCard:
            if card == DevelopmentCard.VICTORY_POINT:
                continue  # cant play VPs
            features[
                f"P{i}_{card.value}_PLAYED"
            ] = player.played_development_cards.count(card)
            features[f"P{i}_NUM_RESOURCES_IN_HAND"] = player.resource_deck.num_cards()
            features[f"P{i}_NUM_DEVS_IN_HAND"] = player.development_deck.num_cards()

    return features


def tile_features(game, p0_color):
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    # TODO: Cacheable
    def f(game, tile_id, resource):
        tile = game.board.get_tile_by_id(tile_id)
        return tile.resource == resource

    # TODO: Cacheable
    def g(game, tile_id):
        tile = game.board.get_tile_by_id(tile_id)
        return 0 if tile.resource is None else number_probability(tile.number)

    def h(game, tile_id):
        tile = game.board.get_tile_by_id(tile_id)
        return game.board.tiles[game.board.robber_coordinate] == tile

    features = {}
    for tile_id in range(NUM_TILES):
        for resource in Resource:
            features[f"TILE{tile_id}_IS_{resource.value}"] = f(game, tile_id, resource)
        features[f"TILE{tile_id}_IS_DESERT"] = f(game, tile_id, None)
        features[f"TILE{tile_id}_PROBA"] = g(game, tile_id)
        features[f"TILE{tile_id}_HAS_ROBBER"] = h(game, tile_id)
    return features


def port_features(game, p0_color):
    # PORT0_WOOD, PORT0_THREE_TO_ONE, ...
    features = {}
    for port_id in range(9):
        for resource in Resource:
            features[f"PORT{port_id}_IS_{resource.value}"] = port_is_resource(
                game, port_id, resource
            )
        features[f"PORT{port_id}_IS_THREE_TO_ONE"] = port_is_threetoone(game, port_id)
    return features


def graph_features(game, p0_color):
    # Features like P0_SETTLEMENT_NODE_1, P0_CITY_NODE_1, ...
    features = {}
    for node_id in range(NUM_NODES):
        for i, player in iter_players(game, p0_color):
            for building in [BuildingType.SETTLEMENT, BuildingType.CITY]:
                features[f"NODE{node_id}_P{i}_{building.value}"] = is_building(
                    game, node_id, player, building
                )
    for edge in get_edges():
        for i, player in iter_players(game, p0_color):
            features[f"EDGE{edge}_P{i}_ROAD"] = is_road(game, edge, player)
    return features


def production_features(game, p0_color):
    # P0_WHEAT_PRODUCTION, P0_ORE_PRODUCTION, ..., P1_WHEAT_PRODUCTION, ...
    features = {}
    for resource in Resource:
        for i, player in iter_players(game, p0_color):
            production = 0
            for node_id in player.buildings[BuildingType.SETTLEMENT]:
                production += get_node_production(game.board, node_id, resource)
            for node_id in player.buildings[BuildingType.CITY]:
                production += 2 * get_node_production(game.board, node_id, resource)
            features[f"P{i}_{resource.value}_PRODUCTION"] = production

    return features


@functools.lru_cache(maxsize=None)
def get_node_production(board, node_id, resource):
    tiles = board.map.get_adjacent_tiles(node_id)
    return sum([number_probability(t.number) for t in tiles if t.resource == resource])


def get_player_expandable_nodes(game, player):
    subgraphs = game.board.find_connected_components(
        player.color
    )  # TODO: Can maintain internally (instead of re-compute).
    enemies = [enemy for enemy in game.state.players if enemy.color != player.color]
    enemy_node_ids = set()
    for enemy in enemies:
        enemy_node_ids.update(enemy.buildings[BuildingType.SETTLEMENT])
        enemy_node_ids.update(enemy.buildings[BuildingType.CITY])

    expandable_node_ids = [
        node_id
        for graph in subgraphs
        for node_id in graph.nodes
        if node_id not in enemy_node_ids  # not plowed
    ]  # not exactly "buildable_node_ids" b.c. we could expand from non-buildable nodes
    return expandable_node_ids


def expansion_features(game, p0_color):
    MAX_EXPANSION_DISTANCE = 4  # exclusive

    features = {}

    # For each connected component node, bfs_edges (skipping enemy edges and nodes nodes)
    empty_edges = set(get_edges())
    for i, player in iter_players(game, p0_color):
        empty_edges.difference_update(player.buildings[BuildingType.ROAD])
    searchable_subgraph = game.board.nxgraph.edge_subgraph(empty_edges)

    board_buildable_node_ids = game.board.buildable_node_ids(
        p0_color, True
    )  # this should be the same for all players. TODO: Can maintain internally (instead of re-compute).

    def skip_blocked_by_enemy(neighbor_ids):
        for node_id in neighbor_ids:
            color = searchable_subgraph.nodes[node_id].get("color", None)
            if color is None or color == player.color:
                yield node_id  # not owned by enemy, can explore

    for i, player in iter_players(game, p0_color):
        expandable_node_ids = get_player_expandable_nodes(game, player)

        # owned_edges = player.buildings[BuildingType.ROAD]
        dis_res_prod = {
            distance: {k: 0 for k in Resource}
            for distance in range(MAX_EXPANSION_DISTANCE)
        }
        for node_id in expandable_node_ids:
            if node_id in board_buildable_node_ids:  # node itself is buildable
                for resource in Resource:
                    production = get_node_production(game.board, node_id, resource)
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
                    production = get_node_production(game.board, b, resource)
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
    ports = game.board.get_port_nodes()
    distances = get_node_distances()
    for resource_or_none in list(Resource) + [None]:
        port_name = "3:1" if resource_or_none is None else resource_or_none.value
        for i, player in iter_players(game, p0_color):
            expandable_node_ids = get_player_expandable_nodes(game, player)
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
    features = {"BANK_DEV_CARDS": game.state.development_deck.num_cards()}
    for resource in Resource:
        features[f"BANK_{resource.value}"] = game.state.resource_deck.count(resource)
    return features


feature_extractors = [
    # PLAYER FEATURES =====
    player_features,
    resource_hand_features,
    # TRANSFERABLE BOARD FEATURES =====
    production_features,
    # expansion_features,
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


def get_feature_ordering():
    global FEATURE_ORDERING
    if FEATURE_ORDERING is None:
        players = [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
        game = Game(players)
        sample = create_sample(game, players[0].color)
        FEATURE_ORDERING = sorted(sample.keys())
    return FEATURE_ORDERING
