"""Graph neural network feature extraction for the Catan board.

Represents the Catan board as a graph where:
- 54 nodes = intersections (vertices) where settlements/cities can be built
- 72 edges = connections between adjacent intersections (where roads go)

Each node carries a feature vector encoding local information:
  buildings, resource production, port access, robber state.
Each edge carries features encoding road ownership.
Global (non-spatial) features are provided separately for concatenation
after the GNN produces a graph-level embedding.

Usage:
    from catanatron.gym.graph_features import (
        build_edge_index,
        build_node_features,
        build_edge_features,
        get_global_features,
        NODE_FEATURE_DIM,
    )

    edge_index = build_edge_index()  # static, compute once
    node_feat = build_node_features(game, p0_color)  # (54, F_node)
    edge_feat = build_edge_features(game, p0_color)  # (144, F_edge)
    global_feat = get_global_features(game, p0_color) # (F_global,)
"""

import functools
import numpy as np
from typing import List, Tuple

from catanatron.game import Game
from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.map import NUM_NODES, number_probability
from catanatron.models.player import Color
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ROAD
from catanatron.state_functions import get_player_buildings
from catanatron.features import iter_players, create_sample
from catanatron.gym.board_tensor_features import is_graph_feature, get_numeric_features


def node_feature_dim(num_players: int) -> int:
    """Total features per node: 2*N + 5 + 6 + 1 + 5 = 2*N + 17."""
    return 2 * num_players + 17


def edge_feature_dim(num_players: int) -> int:
    """One road-ownership channel per player."""
    return num_players


@functools.lru_cache(1)
def build_edge_index() -> np.ndarray:
    """Static COO edge index for PyG / DGL consumption.

    Returns (2, 144) int64 array. Each undirected edge appears in both
    directions so message passing is symmetric. Ordering is stable across
    calls (edges sorted, each pair stored as [a->b, b->a]).
    """
    undirected = sorted(get_edges())
    src, dst = [], []
    for a, b in undirected:
        src.extend([a, b])
        dst.extend([b, a])
    return np.array([src, dst], dtype=np.int64)


@functools.lru_cache(1)
def _sorted_edges() -> List[Tuple[int, int]]:
    return sorted(get_edges())


def build_node_features(game: Game, p0_color: Color) -> np.ndarray:
    """Per-node feature matrix.

    Returns (54, F) float32 array where F = node_feature_dim(num_players).

    Channel layout (all relative to p0_color perspective):
      [0      .. 2N-1 ] Building ownership.
                        Channels 2*i and 2*i+1 correspond to player i
                        (i=0 is the agent). 2*i = settlement (1.0 if present),
                        2*i+1 = city (1.0 if present).
      [2N     .. 2N+4 ] Resource production probability per resource
                        (WOOD, BRICK, SHEEP, WHEAT, ORE). Sums over adjacent
                        tiles, so a node touching two wheat tiles gets the sum.
      [2N+5   .. 2N+10] Port indicators: 5 resource-specific + 1 three-to-one.
      [2N+11           ] Robber-adjacent indicator (any adjacent tile is robbed).
      [2N+12  .. 2N+16] Production blocked by robber, per resource.
    """
    n_players = len(game.state.colors)
    n_feat = node_feature_dim(n_players)
    feat = np.zeros((NUM_NODES, n_feat), dtype=np.float32)

    # Building ownership
    for i, color in iter_players(tuple(game.state.colors), p0_color):
        for nid in get_player_buildings(game.state, color, SETTLEMENT):
            feat[nid, 2 * i] = 1.0
        for nid in get_player_buildings(game.state, color, CITY):
            feat[nid, 2 * i + 1] = 1.0

    # Identify robbed nodes
    catan_map = game.state.board.map
    robber_tile = catan_map.land_tiles.get(game.state.board.robber_coordinate)
    robbed_nodes = set(robber_tile.nodes.values()) if robber_tile else set()

    # Resource production & robber production loss
    col_prod = 2 * n_players  # start of production channels
    col_robber_flag = col_prod + 5 + 6  # robber-adjacent indicator
    col_robber_loss = col_robber_flag + 1  # start of robber loss channels

    for node_id in range(NUM_NODES):
        tiles = catan_map.adjacent_tiles.get(node_id, [])
        for tile in tiles:
            if tile.resource is None:
                continue
            r_idx = RESOURCES.index(tile.resource)
            proba = number_probability(tile.number)
            feat[node_id, col_prod + r_idx] += proba

            if node_id in robbed_nodes:
                feat[node_id, col_robber_loss + r_idx] += proba

    # Robber-adjacent flag
    for node_id in robbed_nodes:
        feat[node_id, col_robber_flag] = 1.0

    # Port indicators
    col_port = col_prod + 5
    for resource, node_ids in catan_map.port_nodes.items():
        p_idx = 5 if resource is None else RESOURCES.index(resource)
        for node_id in node_ids:
            feat[node_id, col_port + p_idx] = 1.0

    return feat


def build_edge_features(game: Game, p0_color: Color) -> np.ndarray:
    """Per-edge feature matrix, aligned with build_edge_index().

    Returns (144, N_players) float32 array. Channel i is 1.0 if player i
    (in p0-relative ordering) owns a road on that edge.
    """
    n_players = len(game.state.colors)
    undirected = _sorted_edges()
    n_directed = len(undirected) * 2
    feat = np.zeros((n_directed, n_players), dtype=np.float32)

    for i, color in iter_players(tuple(game.state.colors), p0_color):
        owned = set()
        for edge in get_player_buildings(game.state, color, ROAD):
            owned.add(tuple(sorted(edge)))
        for idx, (a, b) in enumerate(undirected):
            if tuple(sorted((a, b))) in owned:
                feat[2 * idx, i] = 1.0
                feat[2 * idx + 1, i] = 1.0

    return feat


def get_global_features(game: Game, p0_color: Color) -> np.ndarray:
    """Non-spatial features: player hands, bank, dev cards, game flags.

    These are the features from create_sample() that are NOT graph-spatial
    (i.e. not NODE*, EDGE*, TILE*, PORT* features). They should be
    concatenated with the graph-level readout after the GNN.

    Returns (F_global,) float32 array.
    """
    sample = create_sample(game, p0_color)
    num_players = len(game.state.colors)
    keys = get_numeric_features(num_players)
    return np.array([float(sample[k]) for k in keys], dtype=np.float32)


def global_feature_dim(num_players: int) -> int:
    return len(get_numeric_features(num_players))


def augmented_node_feature_dim(num_players: int) -> int:
    """Node feature dim after pre-aggregating edge info (Strategy B).

    Adds N + 1 channels: per-player road count + unoccupied edge count.
    """
    return node_feature_dim(num_players) + num_players + 1


def augment_node_features_with_edges(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_features: np.ndarray,
    num_players: int,
) -> np.ndarray:
    """Pre-aggregate edge (road) info into per-node features (Strategy B).

    For each node, computes:
      - Per-player count of roads on incident edges
      - Count of unoccupied incident edges

    These are concatenated to the right side of node_features.

    Args:
        node_features: (54, F_node) base node features.
        edge_index: (2, 144) directed edge indices.
        edge_features: (144, N_players) road ownership per directed edge.
        num_players: number of players.

    Returns:
        (54, F_node + 2*N + 1) augmented node features.
    """
    n_nodes = node_features.shape[0]
    extra = np.zeros((n_nodes, num_players + 1), dtype=np.float32)

    src = edge_index[0]
    for i in range(edge_index.shape[1]):
        nid = src[i]
        road_owned = edge_features[i]
        for p in range(num_players):
            extra[nid, p] += road_owned[p]
        if road_owned.sum() == 0:
            extra[nid, num_players] += 1

    return np.concatenate([node_features, extra], axis=1)
