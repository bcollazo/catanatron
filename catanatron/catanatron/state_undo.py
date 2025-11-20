"""
Module for undoing actions on game state.

This allows for efficient tree search by applying and unapplying actions
instead of copying the entire game state.
"""

from typing import Dict, Any
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


def apply_action_undoable(state: State, action: Action) -> tuple[Action, Dict[str, Any]]:
    """
    Applies action and returns undo information.

    Only works for actions in UNDOABLE_ACTIONS.

    Args:
        state: Game state to modify (mutated in place)
        action: Action to apply

    Returns:
        tuple: (fully_specified_action, undo_info) where undo_info contains
               information needed to reverse the action

    Raises:
        ValueError: If action type is not in UNDOABLE_ACTIONS
    """
    if action.action_type not in UNDOABLE_ACTIONS:
        raise ValueError(
            f"Action {action.action_type} is not undoable. "
            f"Only {UNDOABLE_ACTIONS} are currently supported."
        )

    # Save state needed for undo
    undo_info = {
        'previous_playable_actions': state.playable_actions.copy(),
    }

    if action.action_type == ActionType.MARITIME_TRADE:
        trade_offer = action.value
        offering = freqdeck_from_listdeck(
            filter(lambda r: r is not None, trade_offer[:-1])
        )
        asking = freqdeck_from_listdeck(trade_offer[-1:])

        # Store for undo
        undo_info['offering'] = offering
        undo_info['asking'] = asking

        # Apply the trade
        player_freqdeck_subtract(state, action.color, offering)
        state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, offering)
        player_freqdeck_add(state, action.color, asking)
        state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, asking)

        # Note: We don't regenerate playable_actions to save time
        # The caller should regenerate them if needed
        # state.playable_actions = generate_playable_actions(state)

        # Append to action log
        state.actions.append(action)
        undo_info['action_was_appended'] = True

        return action, undo_info

    raise ValueError(f"Unhandled action type: {action.action_type}")


def unapply_action(state: State, action: Action, undo_info: Dict[str, Any]) -> None:
    """
    Reverses the effects of an action using undo information.

    Args:
        state: Game state to modify (mutated in place)
        action: The action that was applied
        undo_info: Undo information returned by apply_action_undoable

    Raises:
        ValueError: If action type is not in UNDOABLE_ACTIONS
    """
    if action.action_type not in UNDOABLE_ACTIONS:
        raise ValueError(
            f"Action {action.action_type} is not undoable. "
            f"Only {UNDOABLE_ACTIONS} are currently supported."
        )

    if action.action_type == ActionType.MARITIME_TRADE:
        offering = undo_info['offering']
        asking = undo_info['asking']

        # Reverse the trade (opposite order of apply)
        state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, asking)
        player_freqdeck_subtract(state, action.color, asking)
        state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, offering)
        player_freqdeck_add(state, action.color, offering)

        # Restore playable actions
        state.playable_actions = undo_info['previous_playable_actions']

        # Remove from action log
        if undo_info.get('action_was_appended', False):
            removed_action = state.actions.pop()
            assert removed_action == action, "Action log mismatch during undo"

        return

    raise ValueError(f"Unhandled action type: {action.action_type}")
