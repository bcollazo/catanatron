from catanatron.models.enums import Action, ActionType
from catanatron.state import State
from catanatron.state_functions import (
    player_clean_turn,
    player_can_play_dev,
    player_deck_replenish,
)
from catanatron.models.player import Color, SimplePlayer, HumanPlayer


def test_playable_cards():
    player = SimplePlayer(Color.RED)

    state = State([player])
    player_deck_replenish(state, Color.RED, "KNIGHT")
    player_clean_turn(state, Color.RED)

    assert player_can_play_dev(state, Color.RED, "KNIGHT")


def test_human_player_asks_for_input():
    # Arrange
    # Create a mock input provider function
    def mock_input_function(prompt):
        return "1"

    player = HumanPlayer(Color.BLUE, input_fn=mock_input_function)

    # Create mock actions
    playable_actions = [
        Action(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.BLUE, ActionType.END_TURN, None),
    ]

    # Act
    chosen_action = player.decide(None, playable_actions)

    # Assert
    assert chosen_action == playable_actions[1]  # Should select the END_TURN action
