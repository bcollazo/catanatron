"""Canonical mapping between CapstoneModel output neurons and
CapstoneCatanatronEnv ACTIONS_ARRAY.

The model produces a 245-dimensional policy logit vector by concatenating
9 policy sub-heads.  The concatenation order in CapstoneModel.forward()
aligns 1:1 with ACTIONS_ARRAY in capstone_env.py, meaning:

    model_output_index  ==  env_action_index

Layout (245 total):
  Indices    Model Head              Env ActionType             Count
  ---------  ----------------------  -------------------------  -----
  [  0.. 71] edge_policy             BUILD_ROAD                   72
  [ 72..125] settlement_vertex       BUILD_SETTLEMENT             54
  [126..179] city_vertex             BUILD_CITY                   54
  [180..198] robber_policy           MOVE_ROBBER                  19
  [199     ] turn_management[0]      DISCARD                       1
  [200..219] trading_policy          MARITIME_TRADE               20
  [220     ] dev_card_policy[0]      BUY_DEVELOPMENT_CARD          1
  [221     ] dev_card_policy[1]      PLAY_KNIGHT_CARD              1
  [222     ] dev_card_policy[2]      PLAY_ROAD_BUILDING            1
  [223..237] yop_resource            PLAY_YEAR_OF_PLENTY          15
  [238..242] monopoly_resource       PLAY_MONOPOLY                 5
  [243     ] turn_management[1]      END_TURN                      1
  [244     ] turn_management[2]      ROLL                          1

turn_management head internal ordering:
  Neuron 0 -> DISCARD,  Neuron 1 -> END_TURN,  Neuron 2 -> ROLL

dev_card_policy head internal ordering:
  Neuron 0 -> BUY_DEVELOPMENT_CARD
  Neuron 1 -> PLAY_KNIGHT_CARD
  Neuron 2 -> PLAY_ROAD_BUILDING
  (Year of Plenty and Monopoly are covered by their own parameter heads)
"""

import numpy as np
from typing import List, Sequence, Tuple

ACTION_SPACE_SIZE = 245


# ── Named slices for each action group ──────────────────────────

ROAD_SLICE       = slice(0, 72)      # 72 edges
SETTLEMENT_SLICE = slice(72, 126)    # 54 nodes
CITY_SLICE       = slice(126, 180)   # 54 nodes
ROBBER_SLICE     = slice(180, 199)   # 19 land tiles
DISCARD_IDX      = 199
MARITIME_SLICE   = slice(200, 220)   # 20 give/take pairs
BUY_DEV_IDX      = 220
KNIGHT_IDX       = 221
ROAD_BUILD_IDX   = 222
YOP_SLICE        = slice(223, 238)   # 15 resource-pair combos
MONOPOLY_SLICE   = slice(238, 243)   # 5 resources
END_TURN_IDX     = 243
ROLL_IDX         = 244


# ── Group metadata (for programmatic iteration) ────────────────

ActionGroup = Tuple[str, str, int, int]  # (env_action_type, model_head, start, stop)

ACTION_GROUPS: List[ActionGroup] = [
    ("BUILD_ROAD",             "edge_policy",        0,   72),
    ("BUILD_SETTLEMENT",       "settlement_vertex",  72,  126),
    ("BUILD_CITY",             "city_vertex",        126, 180),
    ("MOVE_ROBBER",            "robber_policy",      180, 199),
    ("DISCARD",                "turn_management[0]", 199, 200),
    ("MARITIME_TRADE",         "trading_policy",     200, 220),
    ("BUY_DEVELOPMENT_CARD",   "dev_card_policy[0]", 220, 221),
    ("PLAY_KNIGHT_CARD",       "dev_card_policy[1]", 221, 222),
    ("PLAY_ROAD_BUILDING",     "dev_card_policy[2]", 222, 223),
    ("PLAY_YEAR_OF_PLENTY",    "yop_resource",       223, 238),
    ("PLAY_MONOPOLY",          "monopoly_resource",  238, 243),
    ("END_TURN",               "turn_management[1]", 243, 244),
    ("ROLL",                   "turn_management[2]", 244, 245),
]


# ── Mask building ───────────────────────────────────────────────

def make_mask(valid_actions: Sequence[int]) -> np.ndarray:
    """Build a dense binary mask from a sparse list of valid action indices.

    Args:
        valid_actions: Valid action indices (e.g. from
            CapstoneCatanatronEnv.get_valid_actions()).

    Returns:
        np.ndarray of shape (ACTION_SPACE_SIZE,) with 1.0 at valid
        positions and 0.0 everywhere else.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for idx in valid_actions:
        mask[idx] = 1.0
    return mask


# ── Action description (useful for debugging / logging) ─────────

_GROUP_LOOKUP = []
for _atype, _head, _start, _stop in ACTION_GROUPS:
    for _i in range(_start, _stop):
        _GROUP_LOOKUP.append((_atype, _head, _i - _start))
assert len(_GROUP_LOOKUP) == ACTION_SPACE_SIZE


def describe_action(idx: int) -> str:
    """Return a human-readable string for action index *idx*.

    >>> describe_action(0)
    'BUILD_ROAD [edge_policy neuron 0]'
    >>> describe_action(244)
    'ROLL [turn_management[2] neuron 0]'
    """
    if not 0 <= idx < ACTION_SPACE_SIZE:
        raise ValueError(f"Action index {idx} out of range [0, {ACTION_SPACE_SIZE})")
    atype, head, offset = _GROUP_LOOKUP[idx]
    return f"{atype} [{head} neuron {offset}]"


def describe_action_detailed(idx: int) -> str:
    """Like describe_action but resolves the game-engine value via ACTIONS_ARRAY.

    Requires catanatron to be importable.
    """
    from catanatron.gym.envs.capstone_env import ACTIONS_ARRAY
    if not 0 <= idx < ACTION_SPACE_SIZE:
        raise ValueError(f"Action index {idx} out of range [0, {ACTION_SPACE_SIZE})")
    action_type, value = ACTIONS_ARRAY[idx]
    return f"{action_type.value}: {value}"


# ── Validation against the live ACTIONS_ARRAY ───────────────────

def validate():
    """Verify that ACTION_SPACE_SIZE matches the env and that each group's
    ActionType aligns with ACTIONS_ARRAY.  Raises AssertionError on mismatch.

    Call this once at startup (or in tests) to catch layout drift.
    """
    from catanatron.gym.envs.capstone_env import (
        ACTIONS_ARRAY,
        ACTION_SPACE_SIZE as ENV_SIZE,
    )
    from catanatron.models.enums import ActionType

    assert ACTION_SPACE_SIZE == ENV_SIZE, (
        f"action_map.ACTION_SPACE_SIZE ({ACTION_SPACE_SIZE}) != "
        f"capstone_env.ACTION_SPACE_SIZE ({ENV_SIZE})"
    )
    assert ACTION_SPACE_SIZE == len(ACTIONS_ARRAY), (
        f"action_map.ACTION_SPACE_SIZE ({ACTION_SPACE_SIZE}) != "
        f"len(ACTIONS_ARRAY) ({len(ACTIONS_ARRAY)})"
    )

    for atype_name, _head, start, stop in ACTION_GROUPS:
        expected_type = ActionType(atype_name)
        for i in range(start, stop):
            actual_type, _val = ACTIONS_ARRAY[i]
            assert actual_type == expected_type, (
                f"Index {i}: expected {expected_type}, got {actual_type}"
            )

    total = sum(stop - start for _, _, start, stop in ACTION_GROUPS)
    assert total == ACTION_SPACE_SIZE, (
        f"ACTION_GROUPS cover {total} indices, expected {ACTION_SPACE_SIZE}"
    )
