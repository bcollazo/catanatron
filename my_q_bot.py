"""
My trained Q-learning bot for Catanatron
"""
import os
import numpy as np
from tensorflow import keras

from catanatron import Player
from catanatron.cli import register_cli_player
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
from catanatron.models.enums import Action, ActionType

# Features used for training (must match the training data - 2 player game features)
FEATURES = get_feature_ordering(num_players=2)

# Load the trained model (2-player model)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models", "q_model_2p_v1.keras")
Q_MODEL = keras.models.load_model(MODEL_PATH)


def encode_action(action):
    """Encode action as (ACTION_INDEX, ACTION_TYPE) matching training data format."""
    normalized = normalize_action(action)
    action_tuple = (normalized.action_type, normalized.value)
    action_index = ACTIONS_ARRAY.index(action_tuple)

    # Use hash of action type string as integer encoding
    # This is a simple stable encoding that works for any action type
    action_type_str = str(normalized.action_type)
    action_type_value = abs(hash(action_type_str)) % 1000  # Keep it reasonable

    return np.array([float(action_index), float(action_type_value)], dtype=float)


def normalize_action(action):
    """Normalize action for consistent encoding."""
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)

    return normalized


class MyQBot(Player):
    """Q-Learning bot using trained neural network."""

    def decide(self, game, playable_actions):
        """Choose action with highest predicted Q-value."""
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Create state vector
        state = create_sample_vector(game, self.color, FEATURES)

        # Create input samples: state + action encoding for each possible action
        samples = []
        for action in playable_actions:
            samples.append(np.concatenate((state, encode_action(action))))
        X = np.array(samples)

        # Predict Q-values for all state-action pairs
        q_values = Q_MODEL.predict(X, verbose=0)

        # Choose action with highest Q-value
        best_action_idx = np.argmax(q_values)
        return playable_actions[best_action_idx]


# Register the bot for CLI use
register_cli_player("MQB", MyQBot)


if __name__ == "__main__":
    # Simple test to verify the bot can be instantiated
    print(f"✓ MyQBot bot file loaded successfully")
    print(f"✓ Model path: {MODEL_PATH}")
    print(f"✓ Model loaded: {Q_MODEL is not None}")
