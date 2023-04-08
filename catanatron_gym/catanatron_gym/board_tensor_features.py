import networkx as nx
import numpy as np

from catanatron.state_functions import get_player_buildings
from catanatron.models.player import Color
from catanatron.game import Game
from catanatron.models.enums import (
    RESOURCES,
    SETTLEMENT,
    CITY,
    ROAD,
)
from catanatron.models.coordinate_system import offset_to_cube
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.map import number_probability
from catanatron_gym.features import get_feature_ordering, iter_players

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


def is_graph_feature(feature_name):
    return (
        feature_name.startswith("TILE")
        or feature_name.startswith("PORT")
        or feature_name.startswith("NODE")
        or feature_name.startswith("EDGE")
    )


def get_numeric_features(num_players):
    features = get_feature_ordering(num_players)
    return [f for f in features if not is_graph_feature(f)]


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


def create_board_tensor(game: Game, p0_color: Color, channels_first=False):
    """Creates a tensor of shape (WIDTH=21, HEIGHT=11, CHANNELS).

    1 x n hot-encoded planes (2 and 1s for city/settlements).
    1 x n planes for the roads built by each player.
    5 tile resources planes, one per resource.
    1 robber plane (to note nodes blocked by robber).
    6 port planes (one for each resource and one for the 3:1 ports)

    Example:
        - To see WHEAT plane: tf.transpose(board_tensor[:,:,3])
    """
    # add n hot-encoded color multiplier planes (nodes), and n edge planes. 2*n planes
    n = len(game.state.colors)
    channels = 2 * n + 5 + 1 + 6
    planes = [
        [[0.0 for i in range(HEIGHT)] for j in range(WIDTH)] for _ in range(channels)
    ]
    node_map, edge_map = get_node_and_edge_maps()
    for i, color in iter_players(tuple(game.state.colors), p0_color):
        for node_id in get_player_buildings(game.state, color, SETTLEMENT):
            indices = node_map[node_id]
            planes[2 * i][indices[0]][indices[1]] = 1.0
        for node_id in get_player_buildings(game.state, color, CITY):
            indices = node_map[node_id]
            planes[2 * i][indices[0]][indices[1]] = 2.0

        for edge in get_player_buildings(game.state, color, ROAD):
            indices = edge_map[edge]
            planes[2 * i + 1][indices[0]][indices[1]] = 1.0

    # set 5 node-resource probas
    resources = [i for i in RESOURCES]
    tile_map = get_tile_coordinate_map()
    for coordinate, tile in game.state.board.map.land_tiles.items():
        if tile.resource is None:
            continue  # there is already a 3x5 zeros matrix there (everything started as a 0!).

        # Tile looks like:
        # [0.33, 0, 0.33, 0, 0.33]
        # [   0, 0,    0, 0,    0]
        # [0.33, 0, 0.33, 0, 0.33]
        proba = 0 if tile.number is None else number_probability(tile.number)
        (y, x) = tile_map[coordinate]  # returns values in (row, column) math def
        channel_idx = 2 * n + resources.index(tile.resource)
        planes[channel_idx][x][y] += proba
        planes[channel_idx][x + 2][y] += proba
        planes[channel_idx][x + 4][y] += proba
        planes[channel_idx][x][y + 2] += proba
        planes[channel_idx][x + 2][y + 2] += proba
        planes[channel_idx][x + 4][y + 2] += proba

    # set 1 robber channel
    (y, x) = tile_map[game.state.board.robber_coordinate]
    planes[2 * n + 5][x][y] = 1
    planes[2 * n + 5][x + 2][y] = 1
    planes[2 * n + 5][x + 4][y] = 1
    planes[2 * n + 5][x][y + 2] = 1
    planes[2 * n + 5][x + 2][y + 2] = 1
    planes[2 * n + 5][x + 4][y + 2] = 1

    # Q: Would this be simpler as boolean features for each player?
    # add 6 port channels (5 resources + 1 for 3:1 ports)
    # for each port, take index and take node_id coordinates
    for resource, node_ids in game.state.board.map.port_nodes.items():
        channel_idx_delta = 5 if resource is None else resources.index(resource)
        channel_idx = 2 * n + 5 + 1 + channel_idx_delta
        for node_id in node_ids:
            (x, y) = node_map[node_id]
            planes[channel_idx][x][y] = 1

    result = np.array(planes)
    if not channels_first:
        return np.transpose(result, (1, 2, 0))
    return result
