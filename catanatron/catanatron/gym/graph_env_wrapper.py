"""Gymnasium wrapper that produces graph-structured observations.

Wraps CatanatronEnv to output observations as a dictionary of numpy arrays
suitable for GNN consumption (PyG, DGL, or raw PyTorch).

Supports two edge-handling strategies:
  - "direct": Edge features passed as a separate array for edge-aware convs.
  - "preaggregate": Edge info folded into node features; no edge_features in obs.

Usage:
    import gymnasium
    import catanatron.gym
    from catanatron.gym.graph_env_wrapper import GraphObservationWrapper

    base_env = gymnasium.make("catanatron/Catanatron-v0")

    # Strategy A: edge features passed through
    env = GraphObservationWrapper(base_env, edge_strategy="direct")

    # Strategy B: edge info pre-aggregated into node features
    env = GraphObservationWrapper(base_env, edge_strategy="preaggregate")
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.gym.graph_features import (
    build_edge_index,
    build_node_features,
    build_edge_features,
    get_global_features,
    augment_node_features_with_edges,
    node_feature_dim,
    augmented_node_feature_dim,
    edge_feature_dim,
    global_feature_dim,
)
from catanatron.models.map import NUM_NODES


class GraphObservationWrapper(gym.ObservationWrapper):
    """Replaces the default flat/mixed observation with graph-structured data.

    Args:
        env: Base CatanatronEnv (or any wrapper around it).
        edge_strategy: How road/edge info reaches the model.
            "direct"       - obs includes edge_features array (144, N_players).
                             Use with edge-aware convs (GATConv+edge_dim, NNConv).
            "preaggregate" - road counts folded into node_features; no
                             edge_features key. Works with any conv type.
    """

    def __init__(self, env, edge_strategy="direct"):
        super().__init__(env)
        assert edge_strategy in ("direct", "preaggregate")
        self.edge_strategy = edge_strategy

        inner = self.unwrapped
        self._n_players = len(inner.players)

        n_directed_edges = 144
        n_global_feat = global_feature_dim(self._n_players)

        if edge_strategy == "preaggregate":
            n_node_feat = augmented_node_feature_dim(self._n_players)
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(
                    low=0, high=np.inf,
                    shape=(NUM_NODES, n_node_feat),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0, high=NUM_NODES - 1,
                    shape=(2, n_directed_edges),
                    dtype=np.int64,
                ),
                "global_features": spaces.Box(
                    low=0, high=np.inf,
                    shape=(n_global_feat,),
                    dtype=np.float32,
                ),
            })
        else:
            n_node_feat = node_feature_dim(self._n_players)
            n_edge_feat = edge_feature_dim(self._n_players)
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(
                    low=0, high=np.inf,
                    shape=(NUM_NODES, n_node_feat),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0, high=NUM_NODES - 1,
                    shape=(2, n_directed_edges),
                    dtype=np.int64,
                ),
                "edge_features": spaces.Box(
                    low=0, high=1,
                    shape=(n_directed_edges, n_edge_feat),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=0, high=np.inf,
                    shape=(n_global_feat,),
                    dtype=np.float32,
                ),
            })

        self._edge_index = build_edge_index()

    def observation(self, obs):
        inner = self.unwrapped
        game = inner.game
        p0_color = inner.p0.color

        node_feat = build_node_features(game, p0_color)
        edge_index = self._edge_index
        global_feat = get_global_features(game, p0_color)

        if self.edge_strategy == "preaggregate":
            edge_feat = build_edge_features(game, p0_color)
            node_feat = augment_node_features_with_edges(
                node_feat, edge_index, edge_feat, self._n_players
            )
            return {
                "node_features": node_feat,
                "edge_index": edge_index,
                "global_features": global_feat,
            }

        return {
            "node_features": node_feat,
            "edge_index": edge_index,
            "edge_features": build_edge_features(game, p0_color),
            "global_features": global_feat,
        }
