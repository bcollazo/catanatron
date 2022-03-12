import networkx as nx
import tensorflow as tf

from catanatron.state_functions import get_player_buildings
from catanatron.models.player import Color
from catanatron.game import Game
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    VICTORY_POINT,
    SETTLEMENT,
    CITY,
    ROAD,
)
from catanatron.models.coordinate_system import offset_to_cube
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.map import number_probability
from catanatron_gym.features import iter_players

# These assume 4 players
WIDTH = 21
HEIGHT = 11
# CHANNELS = 16  # 4 color multiplier, 5 resource probas, 1 robber, 6 port
# CHANNELS = 9  # 4 color multiplier, 5 resource probas
# CHANNELS = 13  # 8 color multiplier, 5 resource probas
CHANNELS = 20  # 8 color multiplier, 5 resource probas, 1 robber, 6 port


def get_channels(num_players):
    return num_players * 2 + 5 + 1 + 6


NODE_ID_MAP = None
EDGE_MAP = None
TILE_COORDINATE_MAP = None


def get_numeric_features(num_players):
    return sorted(
        set(
            # Player features
            ["P0_ACTUAL_VPS"]
            + [f"P{i}_PUBLIC_VPS" for i in range(num_players)]
            + [f"P{i}_HAS_ARMY" for i in range(num_players)]
            + [f"P{i}_HAS_ROAD" for i in range(num_players)]
            + [f"P{i}_ROADS_LEFT" for i in range(num_players)]
            + [f"P{i}_SETTLEMENTS_LEFT" for i in range(num_players)]
            + [f"P{i}_CITIES_LEFT" for i in range(num_players)]
            + [f"P{i}_HAS_ROLLED" for i in range(num_players)]
            # Player Hand Features
            + [
                f"P{i}_{card}_PLAYED"
                for i in range(num_players)
                for card in DEVELOPMENT_CARDS
                if card != VICTORY_POINT
            ]
            + [f"P{i}_NUM_RESOURCES_IN_HAND" for i in range(num_players)]
            + [f"P{i}_NUM_DEVS_IN_HAND" for i in range(num_players)]
            + [f"P0_{card}_IN_HAND" for card in DEVELOPMENT_CARDS]
            + [
                f"P0_{card}_PLAYABLE"
                for card in DEVELOPMENT_CARDS
                if card != VICTORY_POINT
            ]
            + [f"P0_{resource}_IN_HAND" for resource in RESOURCES]
            # Game Features
            + ["BANK_DEV_CARDS"]
            + [f"BANK_{resource}" for resource in RESOURCES]
        )
    )


NUMERIC_FEATURES = get_numeric_features(4)
NUM_NUMERIC_FEATURES = len(NUMERIC_FEATURES)


def get_node_and_edge_maps():
    global NODE_ID_MAP, EDGE_MAP
    if NODE_ID_MAP is None or EDGE_MAP is None:
        NODE_ID_MAP, EDGE_MAP = init_board_tensor_map()
    return NODE_ID_MAP, EDGE_MAP


def get_tile_coordinate_map():
    global TILE_COORDINATE_MAP
    if TILE_COORDINATE_MAP is None:
        TILE_COORDINATE_MAP = init_tile_coordinate_map()
    return TILE_COORDINATE_MAP


# Create mapping of node_id => i,j and edge => i,j. Respecting (WIDTH, HEIGHT)
def init_board_tensor_map():
    global STATIC_GRAPH
    # These are node-pairs (start,end) for the lines that go from left to right
    pairs = [
        (82, 93),
        (79, 94),
        (42, 25),
        (41, 26),
        (73, 59),
        (72, 60),
    ]
    paths = [nx.shortest_path(STATIC_GRAPH, a, b) for (a, b) in pairs]

    node_map = {}
    edge_map = {}
    for i, path in enumerate(paths):
        for j, node in enumerate(path):
            node_map[node] = (2 * j, 2 * i)

            node_has_down_edge = (i + j) % 2 == 0
            if node_has_down_edge and i + 1 < len(pairs):
                next_path = paths[i + 1]
                edge_map[(node, next_path[j])] = (2 * j, 2 * i + 1)
                edge_map[(next_path[j], node)] = (
                    2 * j,
                    2 * i + 1,
                )

            if j + 1 < len(path):
                edge_map[(node, path[j + 1])] = (2 * j + 1, 2 * i)
                edge_map[(path[j + 1], node)] = (2 * j + 1, 2 * i)

    return node_map, edge_map


def init_tile_coordinate_map():
    """Creates a tile (x,y,z) => i,j mapping,
    where i,j is top-left of 3x6 matrix and respect (WIDTH, HEIGHT) ordering
    """
    tile_map = {}

    width_step = 4  # its really 5, but tiles overlap a column
    height_step = 2  # same here, height is 3, but they overlap a row
    for i in range(HEIGHT // height_step):
        for j in range(WIDTH // width_step):  # +1 b.c. width includes 1/2 water
            (offset_x, offset_y) = (-2 + j, -2 + i)
            cube_coordinate = offset_to_cube((offset_x, offset_y))

            maybe_odd_offset = (i % 2) * 2
            tile_map[cube_coordinate] = (
                height_step * i,
                width_step * j + maybe_odd_offset,
            )
    return tile_map


def create_board_tensor(game: Game, p0_color: Color):
    """Creates a tensor of shape (WIDTH=21, HEIGHT=11, CHANNELS).

    1 x n hot-encoded planes (2 and 1s for city/settlements).
    1 x n planes for the roads built by each player.
    5 tile resources planes, one per resource.
    1 robber plane (to note nodes blocked by robber).
    6 port planes (one for each resource and one for the 3:1 ports)

    Example:
        - To see WHEAT plane: tf.transpose(board_tensor[:,:,3])
    """
    # add 4 hot-encoded color multiplier planes (nodes), and 4 edge planes. 8 planes
    color_multiplier_planes = []
    node_map, edge_map = get_node_and_edge_maps()
    for _, color in iter_players(tuple(game.state.colors), p0_color):
        node_plane = tf.zeros((WIDTH, HEIGHT))
        edge_plane = tf.zeros((WIDTH, HEIGHT))

        indices = []
        updates = []
        for node_id in get_player_buildings(game.state, color, SETTLEMENT):
            indices.append(node_map[node_id])
            updates.append(1)
        for node_id in get_player_buildings(game.state, color, CITY):
            indices.append(node_map[node_id])
            updates.append(2)
        if len(indices) > 0:
            node_plane = tf.tensor_scatter_nd_update(node_plane, indices, updates)

        indices = []
        updates = []
        for edge in get_player_buildings(game.state, color, ROAD):
            indices.append(edge_map[edge])
            updates.append(1)
        if len(indices) > 0:
            edge_plane = tf.tensor_scatter_nd_update(edge_plane, indices, updates)

        color_multiplier_planes.append(node_plane)
        color_multiplier_planes.append(edge_plane)
    color_multiplier_planes = tf.stack(color_multiplier_planes, axis=2)  # axis=channels

    # add 5 node-resource probas, add color edges
    resource_proba_planes = tf.zeros((WIDTH, HEIGHT, 5))
    resources = [i for i in RESOURCES]
    tile_map = get_tile_coordinate_map()
    for (coordinate, tile) in game.state.board.map.land_tiles.items():
        if tile.resource is None:
            continue  # there is already a 3x5 zeros matrix there (everything started as a 0!).

        # Tile looks like:
        # [0.33, 0, 0.33, 0, 0.33]
        # [   0, 0,    0, 0,    0]
        # [0.33, 0, 0.33, 0, 0.33]
        proba = 0 if tile.number is None else number_probability(tile.number)
        (y, x) = tile_map[coordinate]  # returns values in (row, column) math def
        channel_idx = resources.index(tile.resource)
        indices = [[x + i, y + j, channel_idx] for j in range(3) for i in range(5)]
        updates = (
            [proba, 0, proba, 0, proba] + [0, 0, 0, 0, 0] + [proba, 0, proba, 0, proba]
        )
        resource_proba_planes = tf.tensor_scatter_nd_add(
            resource_proba_planes, indices, updates
        )

    # add 1 robber channel
    robber_plane = tf.zeros((WIDTH, HEIGHT, 1))
    (y, x) = tile_map[game.state.board.robber_coordinate]
    indices = [[x + i, y + j, 0] for j in range(3) for i in range(5)]
    updates = [1, 0, 1, 0, 1] + [0, 0, 0, 0, 0] + [1, 0, 1, 0, 1]
    robber_plane = tf.tensor_scatter_nd_add(robber_plane, indices, updates)

    # Q: Would this be simpler as boolean features for each player?
    # add 6 port channels (5 resources + 1 for 3:1 ports)
    # for each port, take index and take node_id coordinates
    port_planes = tf.zeros((WIDTH, HEIGHT, 6))
    for resource, node_ids in game.state.board.map.port_nodes.items():
        channel_idx = 5 if resource is None else resources.index(resource)
        indices = []
        updates = []
        for node_id in node_ids:
            (x, y) = node_map[node_id]
            indices.append([x, y, channel_idx])
            updates.append(1)
        port_planes = tf.tensor_scatter_nd_add(port_planes, indices, updates)

    return tf.concat(
        [color_multiplier_planes, resource_proba_planes, robber_plane, port_planes],
        axis=2,
    )
