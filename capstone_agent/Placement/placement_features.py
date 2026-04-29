"""Compact actor-relative features for the opening placement phase.

The public Capstone observation is a 1259-d vector that includes hands,
strategic summaries, and other main-phase signals.  The placement policy only
needs board-local opening information, so this module exposes a smaller view:

- per node: self/opp occupancy, resource pip totals, port type
- per edge: self/opp road occupancy

The compact representation is intentionally actor-relative so the placement
agent can stay agnostic to absolute player seat identity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from catanatron.gym.envs.capstone_features import get_edge_features, get_vertex_features
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color

try:
    from ..CONSTANTS import EDGE_FEATURE_SLICE, VERTEX_FEATURE_SLICE
except ImportError:  # pragma: no cover - supports script-style imports
    from CONSTANTS import EDGE_FEATURE_SLICE, VERTEX_FEATURE_SLICE


NUM_NODES = 54
NUM_EDGES = 72

CAPSTONE_VERTEX_FEATURE_SIZE = 14
CAPSTONE_EDGE_FEATURE_SIZE = 4

COMPACT_NODE_FEATURE_SIZE = 13
STATIC_NODE_FEATURE_SIZE = 11
COMPACT_EDGE_FEATURE_SIZE = 2
STEP_INDICATOR_SIZE = 4
COMPACT_PLACEMENT_FEATURE_SIZE = (
    NUM_NODES * COMPACT_NODE_FEATURE_SIZE
    + NUM_EDGES * COMPACT_EDGE_FEATURE_SIZE
    + STEP_INDICATOR_SIZE
)

VERTEX_SETTLEMENT_STATUS_IDX = 0
VERTEX_RESOURCE_PIPS_SLICE = slice(1, 6)
VERTEX_PORT_TRADE_SLICE = slice(7, 12)

EDGE_ROAD_STATUS_IDX = 0


def infer_opponent_color(game, self_color: Color) -> Color:
    """Infer the opposing color in the current 1v1 setup."""
    for color in game.state.colors:
        if color != self_color:
            return color
    raise ValueError("Could not infer an opponent color from the game state")


def _port_trade_to_channels(port_trade: np.ndarray) -> np.ndarray:
    """Expand Capstone's 5-d port encoding into 6 explicit channels.

    Capstone encodes:
    - no port: all zeros
    - 3:1 port: all 0.5
    - 2:1 resource port: one-hot 1.0 on the matching resource
    """

    port_trade = np.asarray(port_trade, dtype=np.float32)
    if port_trade.ndim != 2 or port_trade.shape[1] != 5:
        raise ValueError(
            f"Expected a (n, 5) port-trade matrix, got shape {port_trade.shape}"
        )

    is_three_to_one = np.all(np.isclose(port_trade, 0.5), axis=1, keepdims=True)
    specific_ports = np.isclose(port_trade, 1.0).astype(np.float32)
    return np.concatenate([is_three_to_one.astype(np.float32), specific_ports], axis=1)


def _static_node_features_from_vertex_matrix(vertex_matrix: np.ndarray) -> np.ndarray:
    """Extract the actor-independent `(54, 11)` static node block."""

    resource_pips = vertex_matrix[:, VERTEX_RESOURCE_PIPS_SLICE].astype(np.float32)
    port_channels = _port_trade_to_channels(
        vertex_matrix[:, VERTEX_PORT_TRADE_SLICE]
    )
    return np.concatenate([resource_pips, port_channels], axis=1)


def get_static_node_feature_matrix(game) -> np.ndarray:
    """Build the actor-independent `(54, 11)` static node feature matrix."""

    board = game.state.board
    port_node_to_resource = {
        node_id: resource
        for resource, id_set in board.map.port_nodes.items()
        for node_id in id_set
    }

    static_rows = []
    for node_id in range(NUM_NODES):
        pip_counts = board.map.node_production[node_id]
        resource_pips = [pip_counts.get(resource, 0.0) for resource in RESOURCES]

        if node_id not in port_node_to_resource:
            port_trade = np.zeros((1, 5), dtype=np.float32)
        else:
            port_type = port_node_to_resource[node_id]
            if port_type is None:
                port_trade = np.full((1, 5), 0.5, dtype=np.float32)
            else:
                port_trade = np.asarray(
                    [[1.0 if port_type == resource else 0.0 for resource in RESOURCES]],
                    dtype=np.float32,
                )

        port_channels = _port_trade_to_channels(port_trade)[0]
        static_rows.append([*resource_pips, *port_channels.tolist()])

    return np.asarray(static_rows, dtype=np.float32)


def validate_static_node_feature_order(
    game,
    self_color: Optional[Color] = None,
    opp_color: Optional[Color] = None,
):
    """Validate that board-derived node order matches Capstone vertex rows.

    The compact placement contract assumes vertex row `i` corresponds to node id
    `i`. Capstone's live vertex features are built by iterating
    `board.map.node_production.items()`, while offline reconstruction indexes by
    numeric node id directly. This helper checks that those two paths stay aligned.
    """

    if self_color is None:
        self_color = game.state.colors[0]
    if opp_color is None:
        opp_color = infer_opponent_color(game, self_color)

    vertex_matrix = np.asarray(
        get_vertex_features(game, self_color, opp_color), dtype=np.float32
    ).reshape(NUM_NODES, CAPSTONE_VERTEX_FEATURE_SIZE)
    static_from_live_vertices = _static_node_features_from_vertex_matrix(vertex_matrix)
    static_from_board = get_static_node_feature_matrix(game)

    if not np.allclose(static_from_live_vertices, static_from_board, atol=1e-6):
        raise ValueError(
            "Static node feature ordering mismatch between Capstone vertex rows "
            "and board-derived node ids"
        )


def assemble_compact_placement_observation(
    self_settlements: np.ndarray,
    opp_settlements: np.ndarray,
    static_node_features: np.ndarray,
    self_roads: np.ndarray,
    opp_roads: np.ndarray,
) -> np.ndarray:
    """Assemble the canonical compact placement observation layout.

    Layout contract:
    - node block first, one node at a time:
      `[self_settlement, opp_settlement, 5x resource_pips, 6x port_channels]`
    - edge block second, one edge at a time:
      `[self_road, opp_road]`
    - step indicators third:
      `[self_settlement_count, opp_settlement_count, self_road_count, opp_road_count]`
    """

    compact_vertex, step_indicators = assemble_graph_placement_inputs(
        self_settlements,
        opp_settlements,
        static_node_features,
        self_roads,
        opp_roads,
    )
    self_settlements = np.asarray(self_settlements, dtype=np.float32).reshape(NUM_NODES)
    opp_settlements = np.asarray(opp_settlements, dtype=np.float32).reshape(NUM_NODES)
    self_roads = np.asarray(self_roads, dtype=np.float32).reshape(NUM_EDGES)
    opp_roads = np.asarray(opp_roads, dtype=np.float32).reshape(NUM_EDGES)

    compact_edge = np.concatenate(
        [self_roads[:, None], opp_roads[:, None]],
        axis=1,
    )
    return np.concatenate(
        [compact_vertex.reshape(-1), compact_edge.reshape(-1), step_indicators],
        axis=0,
    ).astype(np.float32)


def assemble_graph_placement_inputs(
    self_settlements: np.ndarray,
    opp_settlements: np.ndarray,
    static_node_features: np.ndarray,
    self_roads: np.ndarray,
    opp_roads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return structured graph inputs for the GNN placement model."""

    self_settlements = np.asarray(self_settlements, dtype=np.float32).reshape(NUM_NODES)
    opp_settlements = np.asarray(opp_settlements, dtype=np.float32).reshape(NUM_NODES)
    static_node_features = np.asarray(static_node_features, dtype=np.float32).reshape(
        NUM_NODES, STATIC_NODE_FEATURE_SIZE
    )
    self_roads = np.asarray(self_roads, dtype=np.float32).reshape(NUM_EDGES)
    opp_roads = np.asarray(opp_roads, dtype=np.float32).reshape(NUM_EDGES)

    node_features = np.concatenate(
        [
            self_settlements[:, None],
            opp_settlements[:, None],
            static_node_features,
        ],
        axis=1,
    )
    step_indicators = np.array(
        [
            self_settlements.sum(),
            opp_settlements.sum(),
            self_roads.sum(),
            opp_roads.sum(),
        ],
        dtype=np.float32,
    )
    return node_features, step_indicators


def _project_vertex_matrix(
    vertex_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the 14-d Capstone vertex block to compact node pieces."""

    settlement_status = vertex_matrix[:, [VERTEX_SETTLEMENT_STATUS_IDX]]
    self_settlement = (settlement_status > 0).astype(np.float32).reshape(NUM_NODES)
    opp_settlement = (settlement_status < 0).astype(np.float32).reshape(NUM_NODES)
    static_node_features = _static_node_features_from_vertex_matrix(vertex_matrix)
    return self_settlement, opp_settlement, static_node_features


def _project_edge_matrix(edge_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the 4-d Capstone edge block down to compact road occupancy."""

    road_status = edge_matrix[:, [EDGE_ROAD_STATUS_IDX]]
    self_road = (road_status > 0).astype(np.float32).reshape(NUM_EDGES)
    opp_road = (road_status < 0).astype(np.float32).reshape(NUM_EDGES)
    return self_road, opp_road


def project_capstone_to_compact_placement(
    full_obs: np.ndarray, full_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Project a full Capstone observation into the compact placement view.

    ``full_mask`` is accepted for symmetry with the placement-agent boundary,
    but the compact feature projection depends only on the observation vector.
    """

    del full_mask

    full_obs = np.asarray(full_obs, dtype=np.float32)
    if full_obs.ndim != 1:
        raise ValueError(f"Expected a 1-d observation, got shape {full_obs.shape}")

    vertex_matrix = full_obs[VERTEX_FEATURE_SLICE].reshape(
        NUM_NODES, CAPSTONE_VERTEX_FEATURE_SIZE
    )
    edge_matrix = full_obs[EDGE_FEATURE_SLICE].reshape(
        NUM_EDGES, CAPSTONE_EDGE_FEATURE_SIZE
    )

    self_settlements, opp_settlements, static_node_features = _project_vertex_matrix(
        vertex_matrix
    )
    self_roads, opp_roads = _project_edge_matrix(edge_matrix)
    return assemble_compact_placement_observation(
        self_settlements,
        opp_settlements,
        static_node_features,
        self_roads,
        opp_roads,
    )


def project_capstone_batch_to_compact_placement(full_obs_batch: np.ndarray) -> np.ndarray:
    """Batch version of ``project_capstone_to_compact_placement``."""

    full_obs_batch = np.asarray(full_obs_batch, dtype=np.float32)
    if full_obs_batch.ndim != 2:
        raise ValueError(
            f"Expected a 2-d observation batch, got shape {full_obs_batch.shape}"
        )

    return np.stack(
        [project_capstone_to_compact_placement(obs) for obs in full_obs_batch], axis=0
    ).astype(np.float32)


def get_compact_placement_observation(
    game, self_color: Color, opp_color: Optional[Color] = None
) -> np.ndarray:
    """Build compact placement features directly from a live game object."""

    if opp_color is None:
        opp_color = infer_opponent_color(game, self_color)

    vertex_matrix = np.asarray(
        get_vertex_features(game, self_color, opp_color), dtype=np.float32
    ).reshape(NUM_NODES, CAPSTONE_VERTEX_FEATURE_SIZE)
    edge_matrix = np.asarray(
        get_edge_features(game, self_color, opp_color), dtype=np.float32
    ).reshape(NUM_EDGES, CAPSTONE_EDGE_FEATURE_SIZE)

    self_settlements, opp_settlements, static_node_features = _project_vertex_matrix(
        vertex_matrix
    )
    self_roads, opp_roads = _project_edge_matrix(edge_matrix)
    return assemble_compact_placement_observation(
        self_settlements,
        opp_settlements,
        static_node_features,
        self_roads,
        opp_roads,
    )
