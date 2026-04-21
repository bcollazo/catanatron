"""Compact supervised dataset helpers for the opening placement phase."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterator

import numpy as np

from catanatron.game import GameAccumulator
from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index
from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.enums import ActionType

try:
    from ..CONSTANTS import EDGE_ACTION_SIZE, ROAD_ACTION_SLICE, SETTLEMENT_ACTION_SLICE, VERTEX_ACTION_SIZE
    from .placement_action_space import PlacementPrompt, capstone_action_to_local, local_action_size
    from .placement_features import (
        STATIC_NODE_FEATURE_SIZE,
        assemble_compact_placement_observation,
        get_static_node_feature_matrix,
        validate_static_node_feature_order,
    )
except ImportError:  # pragma: no cover - supports script-style imports
    from CONSTANTS import EDGE_ACTION_SIZE, ROAD_ACTION_SLICE, SETTLEMENT_ACTION_SLICE, VERTEX_ACTION_SIZE
    from placement_action_space import PlacementPrompt, capstone_action_to_local, local_action_size
    from placement_features import (
        STATIC_NODE_FEATURE_SIZE,
        assemble_compact_placement_observation,
        get_static_node_feature_matrix,
        validate_static_node_feature_order,
    )


SCHEMA_VERSION = 1
OPENING_STEP_COUNT = 8
OPENING_ACTION_WIDTH = SETTLEMENT_ACTION_SLICE.stop
OPENING_STEP_TO_ACTOR_SEAT = (0, 0, 1, 1, 1, 1, 0, 0)
OPENING_STEP_TO_PROMPT = (
    PlacementPrompt.SETTLEMENT,
    PlacementPrompt.ROAD,
    PlacementPrompt.SETTLEMENT,
    PlacementPrompt.ROAD,
    PlacementPrompt.SETTLEMENT,
    PlacementPrompt.ROAD,
    PlacementPrompt.SETTLEMENT,
    PlacementPrompt.ROAD,
)

EDGE_ORDER = [tuple(sorted(edge)) for edge in get_edges()]
EDGE_TO_INDEX = {edge: idx for idx, edge in enumerate(EDGE_ORDER)}
LAND_NODE_IDS = frozenset(range(VERTEX_ACTION_SIZE))

REQUIRED_CHUNK_FIELDS = (
    "schema_version",
    "static_node_features",
    "opening_actions_onehot",
    "winner_is_first_actor",
    "game_id",
    "first_actor_color",
    "winner_color",
    "game_seed",
)
REQUIRED_RECORD_FIELDS = REQUIRED_CHUNK_FIELDS[1:]


@dataclass(frozen=True)
class ReconstructedPlacementExample:
    x: np.ndarray
    mask: np.ndarray
    target: int
    prompt: PlacementPrompt
    weight: float
    step_idx: int
    actor_seat: int
    winner_seat: int


def opening_step_actor_seat(step_idx: int) -> int:
    if step_idx < 0 or step_idx >= OPENING_STEP_COUNT:
        raise IndexError(f"Opening step {step_idx} is out of range")
    return OPENING_STEP_TO_ACTOR_SEAT[step_idx]


def opening_step_prompt(step_idx: int) -> PlacementPrompt:
    if step_idx < 0 or step_idx >= OPENING_STEP_COUNT:
        raise IndexError(f"Opening step {step_idx} is out of range")
    return OPENING_STEP_TO_PROMPT[step_idx]


def validate_chunk_schema(loaded) -> int:
    missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in loaded]
    if missing:
        raise ValueError(
            "Compact placement chunk is missing required fields for "
            f"schema v{SCHEMA_VERSION}: {', '.join(missing)}"
        )
    version = int(np.asarray(loaded["schema_version"]).reshape(-1)[0])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported compact placement schema_version={version}; "
            f"expected {SCHEMA_VERSION}"
        )
    return version


def encode_opening_action_onehot(action) -> np.ndarray:
    """Encode an opening settlement/road action into a 126-d one-hot vector."""

    if action.action_type not in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
        raise ValueError(f"Not an opening placement action: {action.action_type}")

    capstone_idx = catanatron_action_to_capstone_index(action)
    if capstone_idx < 0 or capstone_idx >= OPENING_ACTION_WIDTH:
        raise ValueError(
            f"Opening action maps to unexpected Capstone index {capstone_idx}"
        )

    onehot = np.zeros(OPENING_ACTION_WIDTH, dtype=np.float32)
    onehot[capstone_idx] = 1.0
    return onehot


def decode_opening_action_onehot(action_onehot: np.ndarray) -> int:
    """Decode a 126-d opening one-hot vector back to its Capstone index."""

    action_onehot = np.asarray(action_onehot, dtype=np.float32).reshape(OPENING_ACTION_WIDTH)
    active = np.flatnonzero(action_onehot > 0.5)
    if len(active) != 1:
        raise ValueError(
            f"Expected exactly one active opening action bit, got {len(active)}"
        )
    return int(active[0])


def encode_static_node_features(game) -> np.ndarray:
    """Encode the actor-independent `(54, 11)` static node block."""

    validate_static_node_feature_order(game)
    return get_static_node_feature_matrix(game)


def load_chunk_records(path: str) -> list[dict]:
    """Load one compact chunk file into per-game records."""

    records = []
    with np.load(path, allow_pickle=False) as loaded:
        validate_chunk_schema(loaded)

        static_node_features = loaded["static_node_features"]
        opening_actions_onehot = loaded["opening_actions_onehot"]
        winner_is_first_actor = loaded["winner_is_first_actor"]
        game_ids = loaded["game_id"]
        first_actor_colors = loaded["first_actor_color"]
        winner_colors = loaded["winner_color"]
        seeds = loaded["game_seed"]

        for idx in range(len(static_node_features)):
            record = {
                "schema_version": SCHEMA_VERSION,
                "static_node_features": static_node_features[idx],
                "opening_actions_onehot": opening_actions_onehot[idx],
                "winner_is_first_actor": bool(winner_is_first_actor[idx]),
                "game_id": str(game_ids[idx]),
                "first_actor_color": str(first_actor_colors[idx]),
                "winner_color": str(winner_colors[idx]),
                "game_seed": int(seeds[idx]),
            }
            records.append(record)
    return records


def build_chunk_arrays(records: list[dict]) -> dict:
    """Convert per-game records into fixed-shape arrays for one chunk."""

    if len(records) == 0:
        raise ValueError("Cannot build a compact placement chunk from zero records")

    for idx, record in enumerate(records):
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise ValueError(
                "Compact placement record "
                f"{idx} is missing required fields for schema v{SCHEMA_VERSION}: "
                f"{', '.join(missing)}"
            )

    return {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
        "static_node_features": np.asarray(
            [record["static_node_features"] for record in records], dtype=np.float32
        ),
        "opening_actions_onehot": np.asarray(
            [record["opening_actions_onehot"] for record in records], dtype=np.float32
        ),
        "winner_is_first_actor": np.asarray(
            [record["winner_is_first_actor"] for record in records], dtype=np.bool_
        ),
        "game_id": np.asarray([record["game_id"] for record in records]),
        "first_actor_color": np.asarray(
            [record["first_actor_color"] for record in records]
        ),
        "winner_color": np.asarray([record["winner_color"] for record in records]),
        "game_seed": np.asarray([record["game_seed"] for record in records], dtype=np.int64),
    }


def save_chunk_records(path: str, records: list[dict]):
    """Write one compact placement chunk atomically."""

    path = str(path)
    if not path.endswith(".npz"):
        path = f"{path}.npz"

    arrays = build_chunk_arrays(records)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _replay_prior_actions(prior_actions: np.ndarray):
    prior_actions = np.asarray(prior_actions, dtype=np.float32)
    if prior_actions.size == 0:
        prior_actions = prior_actions.reshape(0, OPENING_ACTION_WIDTH)
    else:
        prior_actions = prior_actions.reshape(-1, OPENING_ACTION_WIDTH)

    settlements = np.zeros((2, VERTEX_ACTION_SIZE), dtype=np.float32)
    roads = np.zeros((2, EDGE_ACTION_SIZE), dtype=np.float32)
    last_settlement_by_seat = [-1, -1]

    for step_idx, action_onehot in enumerate(prior_actions):
        prompt = opening_step_prompt(step_idx)
        actor_seat = opening_step_actor_seat(step_idx)
        capstone_idx = decode_opening_action_onehot(action_onehot)
        local_idx = capstone_action_to_local(prompt, capstone_idx)

        if prompt == PlacementPrompt.SETTLEMENT:
            settlements[actor_seat, local_idx] = 1.0
            last_settlement_by_seat[actor_seat] = local_idx
        else:
            roads[actor_seat, local_idx] = 1.0

    return settlements, roads, last_settlement_by_seat


def reconstruct_compact_x(
    static_node_features: np.ndarray,
    prior_actions: np.ndarray,
    step_idx: int,
) -> np.ndarray:
    """Reconstruct the compact placement observation before `step_idx`."""

    settlements, roads, _ = _replay_prior_actions(prior_actions)
    actor_seat = opening_step_actor_seat(step_idx)
    opp_seat = 1 - actor_seat
    return assemble_compact_placement_observation(
        settlements[actor_seat],
        settlements[opp_seat],
        static_node_features,
        roads[actor_seat],
        roads[opp_seat],
    )


def reconstruct_local_mask(prior_actions: np.ndarray, step_idx: int) -> np.ndarray:
    """Reconstruct the legal local placement mask before `step_idx`."""

    settlements, roads, last_settlement_by_seat = _replay_prior_actions(prior_actions)
    occupied_nodes = settlements.sum(axis=0) > 0.5
    occupied_edges = roads.sum(axis=0) > 0.5
    actor_seat = opening_step_actor_seat(step_idx)
    prompt = opening_step_prompt(step_idx)

    if prompt == PlacementPrompt.SETTLEMENT:
        mask = np.zeros(local_action_size(prompt), dtype=np.float32)
        for node_id in range(VERTEX_ACTION_SIZE):
            if occupied_nodes[node_id]:
                continue
            if any(
                neighbor in LAND_NODE_IDS and occupied_nodes[neighbor]
                for neighbor in STATIC_GRAPH.neighbors(node_id)
            ):
                continue
            mask[node_id] = 1.0
        return mask

    settlement_node = last_settlement_by_seat[actor_seat]
    if settlement_node < 0:
        raise ValueError("Cannot reconstruct opening road mask before a settlement exists")

    mask = np.zeros(local_action_size(prompt), dtype=np.float32)
    for neighbor in STATIC_GRAPH.neighbors(settlement_node):
        edge = tuple(sorted((settlement_node, neighbor)))
        if edge not in EDGE_TO_INDEX:
            continue
        edge_idx = EDGE_TO_INDEX[edge]
        if occupied_edges[edge_idx]:
            continue
        mask[edge_idx] = 1.0
    return mask


def extract_local_target(action_onehot: np.ndarray, step_idx: int) -> int:
    """Decode the local action target for the active prompt at `step_idx`."""

    prompt = opening_step_prompt(step_idx)
    capstone_idx = decode_opening_action_onehot(action_onehot)
    return capstone_action_to_local(prompt, capstone_idx)


def iter_reconstructed_examples(
    record: dict,
    selection_mode: str = "winner_only",
    win_weight: float = 1.0,
    loss_weight: float = 0.1,
) -> Iterator[ReconstructedPlacementExample]:
    """Yield supervised placement examples reconstructed from one game record."""

    static_node_features = np.asarray(
        record["static_node_features"], dtype=np.float32
    ).reshape(VERTEX_ACTION_SIZE, STATIC_NODE_FEATURE_SIZE)
    opening_actions = np.asarray(
        record["opening_actions_onehot"], dtype=np.float32
    ).reshape(OPENING_STEP_COUNT, OPENING_ACTION_WIDTH)
    winner_seat = 0 if bool(record["winner_is_first_actor"]) else 1

    if selection_mode not in {"winner_only", "outcome_weighted", "all_examples"}:
        raise ValueError(
            "selection_mode must be one of: winner_only, outcome_weighted, all_examples"
        )

    for step_idx in range(OPENING_STEP_COUNT):
        actor_seat = opening_step_actor_seat(step_idx)
        if selection_mode == "winner_only" and actor_seat != winner_seat:
            continue

        weight = 1.0
        if selection_mode == "outcome_weighted":
            weight = win_weight if actor_seat == winner_seat else loss_weight
        elif selection_mode == "winner_only":
            weight = win_weight

        prior_actions = opening_actions[:step_idx]
        yield ReconstructedPlacementExample(
            x=reconstruct_compact_x(static_node_features, prior_actions, step_idx),
            mask=reconstruct_local_mask(prior_actions, step_idx),
            target=extract_local_target(opening_actions[step_idx], step_idx),
            prompt=opening_step_prompt(step_idx),
            weight=float(weight),
            step_idx=step_idx,
            actor_seat=actor_seat,
            winner_seat=winner_seat,
        )


class CompactPlacementAccumulator(GameAccumulator):
    """Capture one compact per-game opening-placement record."""

    def before(self, game):
        self.static_node_features = encode_static_node_features(game)
        self.first_actor_color = game.state.current_color().value
        self.opening_actions_onehot = []
        self.record = None
        self.skip_reason = None

    def step(self, game_before_action, action):
        if not game_before_action.state.is_initial_build_phase:
            return
        if action.action_type not in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
            return
        self.opening_actions_onehot.append(encode_opening_action_onehot(action))

    def step_after(self, game_after_action, action):
        del game_after_action, action

    def after(self, game):
        winning_color = game.winning_color()
        if winning_color is None:
            self.skip_reason = "no_winner"
            return
        if len(self.opening_actions_onehot) != OPENING_STEP_COUNT:
            self.skip_reason = (
                f"expected {OPENING_STEP_COUNT} opening plies, "
                f"saw {len(self.opening_actions_onehot)}"
            )
            return

        self.record = {
            "schema_version": SCHEMA_VERSION,
            "static_node_features": self.static_node_features.astype(np.float32),
            "opening_actions_onehot": np.asarray(
                self.opening_actions_onehot, dtype=np.float32
            ),
            "winner_is_first_actor": bool(winning_color.value == self.first_actor_color),
            "game_id": str(game.id),
            "first_actor_color": self.first_actor_color,
            "winner_color": winning_color.value,
            "game_seed": int(game.seed),
        }
