"""
Tests for state_undo module - applying and unapplying actions.
"""

import pytest

from catanatron.state import State, apply_action
from catanatron.state_undo import unapply_action, UNDOABLE_ACTIONS
from catanatron.state_functions import player_deck_replenish
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import Action, ActionType, BRICK, ORE, WOOD, WHEAT, SHEEP


def test_maritime_trade_4_to_1_undo():
    """Test applying and unapplying a 4:1 maritime trade."""
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    state.is_initial_build_phase = False

    # Give player resources to trade (4 WOOD for 1 WHEAT)
    player_deck_replenish(state, Color.RED, WOOD, 4)

    # Create the trade action
    trade_action = Action(
        color=Color.RED,
        action_type=ActionType.MARITIME_TRADE,
        value=[WOOD, WOOD, WOOD, WOOD, WHEAT],
    )

    # Save initial state values
    initial_wood = 4
    initial_wheat = 0
    initial_actions_count = len(state.actions)
    initial_history_count = len(state.playable_actions_history)

    # Apply the action
    apply_action(state, trade_action)

    # Verify trade was applied
    from catanatron.state_functions import player_num_resource_cards

    assert player_num_resource_cards(state, Color.RED, WOOD) == 0
    assert player_num_resource_cards(state, Color.RED, WHEAT) == 1
    assert len(state.actions) == initial_actions_count + 1
    assert len(state.playable_actions_history) == initial_history_count + 1

    # Unapply the action
    unapply_action(state, trade_action)

    # Verify state is restored
    assert player_num_resource_cards(state, Color.RED, WOOD) == initial_wood
    assert player_num_resource_cards(state, Color.RED, WHEAT) == initial_wheat
    assert len(state.actions) == initial_actions_count
    assert len(state.playable_actions_history) == initial_history_count


def test_maritime_trade_3_to_1_undo():
    """Test applying and unapplying a 3:1 maritime trade."""
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    state.is_initial_build_phase = False

    # Give player resources to trade (3 BRICK for 1 ORE)
    player_deck_replenish(state, Color.RED, BRICK, 3)

    trade_action = Action(
        color=Color.RED,
        action_type=ActionType.MARITIME_TRADE,
        value=[BRICK, BRICK, BRICK, None, ORE],
    )

    # Save initial state
    initial_brick = 3
    initial_ore = 0

    # Apply and verify
    apply_action(state, trade_action)
    from catanatron.state_functions import player_num_resource_cards

    assert player_num_resource_cards(state, Color.RED, BRICK) == 0
    assert player_num_resource_cards(state, Color.RED, ORE) == 1

    # Unapply and verify
    unapply_action(state, trade_action)
    assert player_num_resource_cards(state, Color.RED, BRICK) == initial_brick
    assert player_num_resource_cards(state, Color.RED, ORE) == initial_ore


def test_multiple_apply_unapply_cycles():
    """Test that multiple apply/unapply cycles work correctly."""
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    state.is_initial_build_phase = False

    # Give player resources
    player_deck_replenish(state, Color.RED, SHEEP, 12)

    # Save initial state
    from catanatron.state_functions import player_num_resource_cards

    initial_sheep = player_num_resource_cards(state, Color.RED, SHEEP)
    initial_wheat = player_num_resource_cards(state, Color.RED, WHEAT)
    initial_actions_count = len(state.actions)
    initial_history_count = len(state.playable_actions_history)

    # Perform 3 apply/unapply cycles
    for i in range(3):
        trade_action = Action(
            color=Color.RED,
            action_type=ActionType.MARITIME_TRADE,
            value=[SHEEP, SHEEP, SHEEP, SHEEP, WHEAT],
        )

        apply_action(state, trade_action)
        assert player_num_resource_cards(state, Color.RED, SHEEP) == initial_sheep - 4
        assert player_num_resource_cards(state, Color.RED, WHEAT) == initial_wheat + 1

        unapply_action(state, trade_action)
        assert player_num_resource_cards(state, Color.RED, SHEEP) == initial_sheep
        assert player_num_resource_cards(state, Color.RED, WHEAT) == initial_wheat
        assert len(state.actions) == initial_actions_count
        assert len(state.playable_actions_history) == initial_history_count


def test_undoable_actions_set():
    """Verify that UNDOABLE_ACTIONS contains expected action types."""
    assert ActionType.MARITIME_TRADE in UNDOABLE_ACTIONS
    # As we add more undoable actions, add assertions here


def test_non_undoable_action_raises_error():
    """Test that attempting to undo a non-undoable action raises ValueError."""
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    # Try to unapply an action that's not in UNDOABLE_ACTIONS
    non_undoable_action = Action(
        color=Color.RED, action_type=ActionType.ROLL, value=(3, 4)
    )

    with pytest.raises(ValueError, match="is not undoable"):
        unapply_action(state, non_undoable_action)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
