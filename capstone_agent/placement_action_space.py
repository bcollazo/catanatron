"""Helpers for a prompt-specific opening placement action space."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from catanatron.models.enums import ActionPrompt

try:
    from .CONSTANTS import ROAD_ACTION_SLICE, SETTLEMENT_ACTION_SLICE
except ImportError:  # pragma: no cover - supports script-style imports
    from CONSTANTS import ROAD_ACTION_SLICE, SETTLEMENT_ACTION_SLICE


class PlacementPrompt(IntEnum):
    SETTLEMENT = 0
    ROAD = 1


def prompt_name(prompt: PlacementPrompt) -> str:
    return "settlement" if prompt == PlacementPrompt.SETTLEMENT else "road"


def local_action_size(prompt: PlacementPrompt) -> int:
    if prompt == PlacementPrompt.SETTLEMENT:
        return SETTLEMENT_ACTION_SLICE.stop - SETTLEMENT_ACTION_SLICE.start
    if prompt == PlacementPrompt.ROAD:
        return ROAD_ACTION_SLICE.stop - ROAD_ACTION_SLICE.start
    raise ValueError(f"Unknown placement prompt: {prompt}")


def infer_placement_prompt(full_mask: np.ndarray) -> PlacementPrompt:
    """Infer whether the current placement decision is settlement or road."""

    full_mask = np.asarray(full_mask)
    settlement_valid = np.any(full_mask[SETTLEMENT_ACTION_SLICE] > 0.5)
    road_valid = np.any(full_mask[ROAD_ACTION_SLICE] > 0.5)

    if settlement_valid and not road_valid:
        return PlacementPrompt.SETTLEMENT
    if road_valid and not settlement_valid:
        return PlacementPrompt.ROAD
    if settlement_valid and road_valid:
        raise ValueError(
            "Placement mask is ambiguous: both settlement and road actions are valid"
        )
    raise ValueError("Placement mask contains no valid settlement or road actions")


def prompt_from_game_state(state) -> PlacementPrompt:
    """Read the placement prompt directly from the live game state."""

    if state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return PlacementPrompt.SETTLEMENT
    if state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        return PlacementPrompt.ROAD
    raise ValueError(f"State is not in an opening placement prompt: {state.current_prompt}")


def prompt_from_game(game) -> PlacementPrompt:
    return prompt_from_game_state(game.state)


def capstone_mask_to_local_mask(
    full_mask: np.ndarray, prompt: PlacementPrompt
) -> np.ndarray:
    """Project the full 245-d Capstone mask into the active local head mask."""

    full_mask = np.asarray(full_mask, dtype=np.float32)
    if prompt == PlacementPrompt.SETTLEMENT:
        return full_mask[SETTLEMENT_ACTION_SLICE].copy()
    if prompt == PlacementPrompt.ROAD:
        return full_mask[ROAD_ACTION_SLICE].copy()
    raise ValueError(f"Unknown placement prompt: {prompt}")


def local_action_to_capstone(prompt: PlacementPrompt, local_idx: int) -> int:
    """Map a local placement action index back into the Capstone action space."""

    local_idx = int(local_idx)
    size = local_action_size(prompt)
    if local_idx < 0 or local_idx >= size:
        raise ValueError(
            f"Local action {local_idx} is out of range for {prompt_name(prompt)}"
        )

    if prompt == PlacementPrompt.SETTLEMENT:
        return SETTLEMENT_ACTION_SLICE.start + local_idx
    if prompt == PlacementPrompt.ROAD:
        return ROAD_ACTION_SLICE.start + local_idx
    raise ValueError(f"Unknown placement prompt: {prompt}")


def capstone_action_to_local(prompt: PlacementPrompt, capstone_idx: int) -> int:
    """Map a global Capstone action index into the active local head index."""

    capstone_idx = int(capstone_idx)
    if prompt == PlacementPrompt.SETTLEMENT:
        if capstone_idx not in range(
            SETTLEMENT_ACTION_SLICE.start, SETTLEMENT_ACTION_SLICE.stop
        ):
            raise ValueError(
                f"Capstone action {capstone_idx} is not a settlement action"
            )
        return capstone_idx - SETTLEMENT_ACTION_SLICE.start

    if prompt == PlacementPrompt.ROAD:
        if capstone_idx not in range(ROAD_ACTION_SLICE.start, ROAD_ACTION_SLICE.stop):
            raise ValueError(f"Capstone action {capstone_idx} is not a road action")
        return capstone_idx - ROAD_ACTION_SLICE.start

    raise ValueError(f"Unknown placement prompt: {prompt}")
