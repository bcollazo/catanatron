"""Bidirectional translator between Catanatron and Capstone action spaces.

Catanatron defines a 290-action space; Capstone defines a 245-action space.
Both wrap the same game engine but order actions differently and represent
maritime trades at different granularity (Catanatron encodes trade rate in
a 5-tuple; Capstone uses a simplified (give, take) pair).

Public API:
    catanatron_to_capstone(cat_idx) -> cap_idx
    capstone_to_catanatron(cap_idx, playable_actions) -> cat_idx
    capstone_to_action(cap_idx, playable_actions) -> Action
    capstone_to_catanatron_from_state(cap_idx, game, color) -> cat_idx
    catanatron_action_to_capstone_index(action) -> cap_idx
    batch_catanatron_to_capstone(cat_indices) -> list[cap_idx]
"""

from typing import Dict, List, Set, Tuple

from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.gym.envs.catanatron_env import (
    ACTIONS_ARRAY as CAT_ACTIONS,
    normalize_action as cat_normalize_action,
)
from catanatron.gym.envs.capstone_env import (
    ACTIONS_ARRAY as CAP_ACTIONS,
)

CAT_ACTION_SPACE_SIZE = len(CAT_ACTIONS)
CAP_ACTION_SPACE_SIZE = len(CAP_ACTIONS)


# ── helpers ──────────────────────────────────────────────────────

def _extract_maritime_give_take(value_5tuple) -> Tuple[str, str]:
    """Extract (give_resource, take_resource) from a Catanatron 5-element
    maritime trade tuple like ('WOOD','WOOD','WOOD','WOOD','BRICK')."""
    return (value_5tuple[0], value_5tuple[4])


def _build_maritime_5tuple(give: str, take: str, rate: int) -> tuple:
    """Construct a Catanatron-format 5-tuple for a maritime trade at *rate*:1."""
    padding = [None] * (4 - rate)
    return tuple([give] * rate + padding + [take])


def _to_capstone_action_key_from_catan_action(action: Action) -> tuple:
    """Normalize a Catanatron Action to the Capstone action-space key."""
    normalized = cat_normalize_action(action)
    atype = normalized.action_type
    value = normalized.value

    if atype == ActionType.MARITIME_TRADE and isinstance(value, tuple):
        value = _extract_maritime_give_take(value)
    elif (
        atype == ActionType.PLAY_YEAR_OF_PLENTY
        and isinstance(value, tuple)
        and len(value) == 1
    ):
        # Capstone does not encode one-card YOP; map to (r, r).
        value = (value[0], value[0])

    return (atype, value)


# ── build static lookup tables at import time ────────────────────

_CAP_VALUE_TO_IDX: Dict[tuple, int] = {
    entry: idx for idx, entry in enumerate(CAP_ACTIONS)
}

_CAT_VALUE_TO_IDX: Dict[tuple, int] = {
    entry: idx for idx, entry in enumerate(CAT_ACTIONS)
}

_CAT_TO_CAP: Dict[int, int] = {}
for _cat_idx, (_atype, _val) in enumerate(CAT_ACTIONS):
    if _atype == ActionType.MARITIME_TRADE:
        _give, _take = _extract_maritime_give_take(_val)
        _CAT_TO_CAP[_cat_idx] = _CAP_VALUE_TO_IDX[
            (ActionType.MARITIME_TRADE, (_give, _take))
        ]
    elif (
        _atype == ActionType.PLAY_YEAR_OF_PLENTY
        and isinstance(_val, tuple)
        and len(_val) == 1
    ):
        _CAT_TO_CAP[_cat_idx] = _CAP_VALUE_TO_IDX[
            (ActionType.PLAY_YEAR_OF_PLENTY, (_val[0], _val[0]))
        ]
    else:
        _CAT_TO_CAP[_cat_idx] = _CAP_VALUE_TO_IDX[(_atype, _val)]

_CAP_TO_CAT: Dict[int, int] = {}
_CAP_MARITIME_INDICES: Set[int] = set()
for _cap_idx, (_atype, _val) in enumerate(CAP_ACTIONS):
    if _atype == ActionType.MARITIME_TRADE:
        _CAP_MARITIME_INDICES.add(_cap_idx)
    else:
        _CAP_TO_CAT[_cap_idx] = _CAT_VALUE_TO_IDX[(_atype, _val)]

# cleanup module-level loop vars
del _cat_idx, _cap_idx, _atype, _val, _give, _take


# ── public API ───────────────────────────────────────────────────

def catanatron_to_capstone(cat_idx: int) -> int:
    """Map a Catanatron action index to its Capstone equivalent.

    Pure function — no game state required.
    Maritime trade actions (3 Catanatron variants per resource pair) collapse
    to a single Capstone index.  Single-card Year of Plenty maps to the
    double-card variant of the same resource.
    """
    if cat_idx not in _CAT_TO_CAP:
        raise KeyError(f"Invalid Catanatron action index: {cat_idx}")
    return _CAT_TO_CAP[cat_idx]


def capstone_to_catanatron(cap_idx: int, playable_actions: List[Action]) -> int:
    """Map a Capstone action index to its Catanatron equivalent.

    For maritime trades, scans *playable_actions* to determine the correct
    trade rate (4:1, 3:1, or 2:1).  The game engine only generates the single
    valid rate for each resource, so this is unambiguous.
    """
    if cap_idx < 0 or cap_idx >= CAP_ACTION_SPACE_SIZE:
        raise KeyError(f"Invalid Capstone action index: {cap_idx}")

    desired_key = CAP_ACTIONS[cap_idx]
    for action in playable_actions:
        if _to_capstone_action_key_from_catan_action(action) == desired_key:
            normalized = cat_normalize_action(action)
            cat_key = (normalized.action_type, normalized.value)
            return _CAT_VALUE_TO_IDX[cat_key]

    # Fallback static remap for deterministic non-ambiguous actions.
    if cap_idx in _CAP_TO_CAT:
        return _CAP_TO_CAT[cap_idx]

    # Maritime can be ambiguous without playable-actions context.
    if CAP_ACTIONS[cap_idx][0] == ActionType.MARITIME_TRADE:
        give, take = CAP_ACTIONS[cap_idx][1]
        raise ValueError(
            "No playable maritime trade matches "
            f"give={give}, take={take}"
        )

    raise ValueError(
        f"No playable action matches Capstone action index {cap_idx}"
    )


def capstone_to_action(cap_idx: int, playable_actions: List[Action]) -> Action:
    """Map a Capstone action index directly to an engine Action object.

    Single-pass alternative to capstone_to_catanatron() followed by
    catanatron_env.from_action_space().  Scans *playable_actions* once
    and returns the matching Action without an intermediate index.
    """
    if cap_idx < 0 or cap_idx >= CAP_ACTION_SPACE_SIZE:
        raise KeyError(f"Invalid Capstone action index: {cap_idx}")

    desired_key = CAP_ACTIONS[cap_idx]
    for action in playable_actions:
        if _to_capstone_action_key_from_catan_action(action) == desired_key:
            return action

    raise ValueError(
        f"No playable action matches Capstone action index {cap_idx}"
    )


def capstone_to_catanatron_from_state(cap_idx: int, game, color) -> int:
    """Map a Capstone action index to its Catanatron equivalent using game state.

    For maritime trades, determines the trade rate from the player's ports
    instead of scanning playable_actions.
    """
    if cap_idx not in _CAP_MARITIME_INDICES:
        if cap_idx not in _CAP_TO_CAT:
            raise KeyError(f"Invalid Capstone action index: {cap_idx}")
        return _CAP_TO_CAT[cap_idx]

    give, take = CAP_ACTIONS[cap_idx][1]

    port_resources = game.state.board.get_player_port_resources(color)
    rate = 4
    if None in port_resources:
        rate = 3
    if give in port_resources:
        rate = 2

    cat_value = _build_maritime_5tuple(give, take, rate)
    return _CAT_VALUE_TO_IDX[(ActionType.MARITIME_TRADE, cat_value)]


def catanatron_action_to_capstone_index(action: Action) -> int:
    """Convert a raw Catanatron Action object to a Capstone action index.

    Normalises the action (sorting road edges, stripping roll values, etc.)
    then looks up through the Catanatron table into the Capstone table.
    """
    normalized = cat_normalize_action(action)
    cat_key = (normalized.action_type, normalized.value)
    cat_idx = _CAT_VALUE_TO_IDX[cat_key]
    return _CAT_TO_CAP[cat_idx]


def batch_catanatron_to_capstone(cat_indices: List[int]) -> List[int]:
    """Convert a list of Catanatron action indices to Capstone indices."""
    return [_CAT_TO_CAP[i] for i in cat_indices]


# ── self-check (runs once at import) ────────────────────────────

def _verify_tables():
    assert len(_CAT_TO_CAP) == CAT_ACTION_SPACE_SIZE, (
        f"Forward table has {len(_CAT_TO_CAP)} entries, "
        f"expected {CAT_ACTION_SPACE_SIZE}"
    )
    non_maritime_cap = CAP_ACTION_SPACE_SIZE - len(_CAP_MARITIME_INDICES)
    assert len(_CAP_TO_CAT) == non_maritime_cap, (
        f"Reverse table has {len(_CAP_TO_CAT)} entries, "
        f"expected {non_maritime_cap}"
    )

    for cat_idx in range(CAT_ACTION_SPACE_SIZE):
        cap_idx = _CAT_TO_CAP[cat_idx]
        assert 0 <= cap_idx < CAP_ACTION_SPACE_SIZE

    for cap_idx in range(CAP_ACTION_SPACE_SIZE):
        if cap_idx in _CAP_MARITIME_INDICES:
            continue
        cat_idx = _CAP_TO_CAT[cap_idx]
        assert 0 <= cat_idx < CAT_ACTION_SPACE_SIZE
        assert _CAT_TO_CAP[cat_idx] == cap_idx, (
            f"Round-trip failed: cap {cap_idx} -> cat {cat_idx} -> "
            f"cap {_CAT_TO_CAP[cat_idx]}"
        )


_verify_tables()
