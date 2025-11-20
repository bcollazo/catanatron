"""
Module for undoing actions on game state.

This allows for efficient tree search by applying and unapplying actions
instead of copying the entire game state.

Usage:
    from catanatron.state import apply_action
    from catanatron.state_undo import unapply_action, UNDOABLE_ACTIONS

    if action.action_type in UNDOABLE_ACTIONS:
        apply_action(state, action)  # normal apply
        # ... do something ...
        unapply_action(state, action)  # reverse it
"""

from catanatron.models.enums import Action, ActionType
from catanatron.state import State
from catanatron.models.decks import (
    freqdeck_from_listdeck,
    freqdeck_add,
    freqdeck_subtract,
)
from catanatron.state_functions import (
    player_freqdeck_add,
    player_freqdeck_subtract,
)


# Set of action types that have undo logic implemented
UNDOABLE_ACTIONS = {
    ActionType.MARITIME_TRADE,
}


def unapply_action(state: State, action: Action) -> None:
    """
    Reverses the effects of an action that was applied with apply_action.

    Uses state.playable_actions_history to restore playable_actions and derives
    the action details from action.value. Works for actions in UNDOABLE_ACTIONS.

    Args:
        state: Game state to modify (mutated in place)
        action: The action that was applied

    Raises:
        ValueError: If action type is not in UNDOABLE_ACTIONS
        AssertionError: If state.actions or playable_actions_history are inconsistent
    """
    if action.action_type not in UNDOABLE_ACTIONS:
        raise ValueError(
            f"Action {action.action_type} is not undoable. "
            f"Only {UNDOABLE_ACTIONS} are currently supported."
        )

    if action.action_type == ActionType.MARITIME_TRADE:
        # Derive trade details from action.value (no need for external undo_info)
        trade_offer = action.value
        offering = freqdeck_from_listdeck(
            filter(lambda r: r is not None, trade_offer[:-1])
        )
        asking = freqdeck_from_listdeck(trade_offer[-1:])

        # Reverse the trade (opposite order of apply)
        state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, asking)
        player_freqdeck_subtract(state, action.color, asking)
        state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, offering)
        player_freqdeck_add(state, action.color, offering)

        # Restore playable_actions from history stack
        assert (
            len(state.playable_actions_history) > 0
        ), "playable_actions_history is empty, cannot undo"
        state.playable_actions = state.playable_actions_history.pop()

        # Remove from action log
        assert len(state.actions) > 0, "actions list is empty, cannot undo"
        removed_action = state.actions.pop()
        assert (
            removed_action == action
        ), f"Action log mismatch during undo: expected {action}, got {removed_action}"

        return

    raise ValueError(f"Unhandled action type: {action.action_type}")
