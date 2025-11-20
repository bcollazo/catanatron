"""
Tests for state_undo module - applying and unapplying actions.
"""

import pytest
from catanatron.game import Game
from catanatron.models.player import Player, Color
from catanatron.models.enums import Action, ActionType
from catanatron.state_undo import (
    apply_action_undoable,
    unapply_action,
    UNDOABLE_ACTIONS,
)
from catanatron.state_functions import player_key


def assert_states_equal(state1, state2, message="States should be equal"):
    """
    Helper function to assert two states are equal.
    Compares all relevant state attributes.
    """
    # Compare player state (resources, victory points, etc.)
    assert state1.player_state == state2.player_state, f"{message}: player_state mismatch"

    # Compare resource and development card decks
    assert state1.resource_freqdeck == state2.resource_freqdeck, f"{message}: resource_freqdeck mismatch"
    assert state1.development_listdeck == state2.development_listdeck, f"{message}: development_listdeck mismatch"

    # Compare turn and player indices
    assert state1.current_player_index == state2.current_player_index, f"{message}: current_player_index mismatch"
    assert state1.current_turn_index == state2.current_turn_index, f"{message}: current_turn_index mismatch"
    assert state1.num_turns == state2.num_turns, f"{message}: num_turns mismatch"

    # Compare prompts and flags
    assert state1.current_prompt == state2.current_prompt, f"{message}: current_prompt mismatch"
    assert state1.is_initial_build_phase == state2.is_initial_build_phase, f"{message}: is_initial_build_phase mismatch"
    assert state1.is_discarding == state2.is_discarding, f"{message}: is_discarding mismatch"
    assert state1.is_moving_knight == state2.is_moving_knight, f"{message}: is_moving_knight mismatch"
    assert state1.is_road_building == state2.is_road_building, f"{message}: is_road_building mismatch"
    assert state1.free_roads_available == state2.free_roads_available, f"{message}: free_roads_available mismatch"

    # Compare actions log length (content might differ for the last action)
    assert len(state1.actions) == len(state2.actions), f"{message}: actions list length mismatch"

    # Compare playable_actions_history
    assert len(state1.playable_actions_history) == len(state2.playable_actions_history), f"{message}: playable_actions_history length mismatch"

    # Compare playable actions
    assert set(state1.playable_actions) == set(state2.playable_actions), f"{message}: playable_actions mismatch"

    # Compare board state
    assert state1.board.buildings == state2.board.buildings, f"{message}: buildings mismatch"
    assert state1.board.roads == state2.board.roads, f"{message}: roads mismatch"
    assert state1.board.robber_coordinate == state2.board.robber_coordinate, f"{message}: robber_coordinate mismatch"

    # Compare buildings by color
    assert state1.buildings_by_color == state2.buildings_by_color, f"{message}: buildings_by_color mismatch"


class TestMaritimeTradeUndo:
    """Tests for undoing MARITIME_TRADE actions."""

    def test_maritime_trade_4_to_1(self):
        """Test applying and unapplying a 4:1 maritime trade."""
        # Create a game and advance to a state where a player can trade
        players = [
            Player(Color.RED),
            Player(Color.BLUE),
        ]
        game = Game(players, seed=42)

        # Play through initial placement to get to normal gameplay
        while game.state.is_initial_build_phase:
            playable_actions = game.state.playable_actions
            game.execute(playable_actions[0])

        # Give the current player resources to trade
        # 4 WOOD to trade for 1 WHEAT
        color = game.state.current_color()
        key = player_key(game.state, color)
        game.state.player_state[f"{key}_WOOD_IN_HAND"] = 4
        game.state.player_state[f"{key}_HAS_ROLLED"] = True

        # Create a 4:1 trade action (4 WOOD for 1 WHEAT)
        # trade_offer format: [res1, res2, res3, res4, asking_resource]
        # Resources are: WOOD, BRICK, SHEEP, WHEAT, ORE
        trade_action = Action(
            color=color,
            action_type=ActionType.MARITIME_TRADE,
            value=["WOOD", "WOOD", "WOOD", "WOOD", "WHEAT"]
        )

        # Save the state before applying
        state_before = game.state.copy()

        # Apply the action
        fully_specified_action = apply_action_undoable(game.state, trade_action)

        # Verify the trade was applied
        assert game.state.player_state[f"{key}_WOOD_IN_HAND"] == 0, "WOOD should be traded away"
        assert game.state.player_state[f"{key}_WHEAT_IN_HAND"] == 1, "WHEAT should be received"

        # Unapply the action
        unapply_action(game.state, fully_specified_action)

        # Verify state is restored
        assert_states_equal(game.state, state_before, "State after unapply should match original")

        # Verify specific resource counts are restored
        assert game.state.player_state[f"{key}_WOOD_IN_HAND"] == 4, "WOOD should be restored"
        assert game.state.player_state[f"{key}_WHEAT_IN_HAND"] == 0, "WHEAT should be restored"

    def test_maritime_trade_3_to_1_with_port(self):
        """Test applying and unapplying a 3:1 maritime trade."""
        players = [
            Player(Color.RED),
            Player(Color.BLUE),
        ]
        game = Game(players, seed=42)

        # Play through initial placement
        while game.state.is_initial_build_phase:
            playable_actions = game.state.playable_actions
            game.execute(playable_actions[0])

        # Give the current player resources to trade
        # 3 BRICK to trade for 1 ORE
        color = game.state.current_color()
        key = player_key(game.state, color)

        # Clear existing resources and set specific amounts
        game.state.player_state[f"{key}_BRICK_IN_HAND"] = 3
        game.state.player_state[f"{key}_ORE_IN_HAND"] = 0
        game.state.player_state[f"{key}_HAS_ROLLED"] = True

        trade_action = Action(
            color=color,
            action_type=ActionType.MARITIME_TRADE,
            value=["BRICK", "BRICK", "BRICK", None, "ORE"]
        )

        # Save the state
        state_before = game.state.copy()

        # Apply and unapply
        fully_specified_action = apply_action_undoable(game.state, trade_action)

        # Verify trade was applied
        assert game.state.player_state[f"{key}_BRICK_IN_HAND"] == 0
        assert game.state.player_state[f"{key}_ORE_IN_HAND"] == 1

        # Unapply
        unapply_action(game.state, fully_specified_action)

        # Verify restoration
        assert_states_equal(game.state, state_before, "State should be restored after 3:1 trade undo")
        assert game.state.player_state[f"{key}_BRICK_IN_HAND"] == 3
        assert game.state.player_state[f"{key}_ORE_IN_HAND"] == 0

    def test_multiple_apply_unapply_cycles(self):
        """Test that multiple apply/unapply cycles work correctly."""
        players = [
            Player(Color.RED),
            Player(Color.BLUE),
        ]
        game = Game(players, seed=42)

        # Play through initial placement
        while game.state.is_initial_build_phase:
            playable_actions = game.state.playable_actions
            game.execute(playable_actions[0])

        color = game.state.current_color()
        key = player_key(game.state, color)
        game.state.player_state[f"{key}_SHEEP_IN_HAND"] = 12
        game.state.player_state[f"{key}_HAS_ROLLED"] = True

        # Save initial state
        initial_state = game.state.copy()

        # Perform 3 apply/unapply cycles
        for i in range(3):
            trade_action = Action(
                color=color,
                action_type=ActionType.MARITIME_TRADE,
                value=["SHEEP", "SHEEP", "SHEEP", "SHEEP", "WHEAT"]
            )

            action = apply_action_undoable(game.state, trade_action)
            assert game.state.player_state[f"{key}_SHEEP_IN_HAND"] == 12 - 4
            assert game.state.player_state[f"{key}_WHEAT_IN_HAND"] == 1

            unapply_action(game.state, action)
            assert_states_equal(game.state, initial_state, f"State should be restored after cycle {i+1}")

    def test_undoable_actions_set(self):
        """Verify that UNDOABLE_ACTIONS contains expected action types."""
        assert ActionType.MARITIME_TRADE in UNDOABLE_ACTIONS
        # As we add more undoable actions, add assertions here

    def test_non_undoable_action_raises_error(self):
        """Test that attempting to undo a non-undoable action raises ValueError."""
        players = [Player(Color.RED), Player(Color.BLUE)]
        game = Game(players, seed=42)

        # Try to apply an action that's not in UNDOABLE_ACTIONS
        non_undoable_action = Action(
            color=Color.RED,
            action_type=ActionType.ROLL,
            value=None
        )

        with pytest.raises(ValueError, match="is not undoable"):
            apply_action_undoable(game.state, non_undoable_action)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
