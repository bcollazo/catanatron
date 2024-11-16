from typing import Iterable
import numpy as np
import os
from sb3_contrib import MaskablePPO

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import from_action_space, to_action_space
from catanatron_gym.features import create_sample, get_feature_ordering
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    is_graph_feature,
)
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN


class PPOPlayer(Player):
    """
    Proximal Policy Optimization (PPO) reinforcement learning agent.
    """

    def __init__(self, color, model_path=None):
        super().__init__(color)
        self.model = None
        self.numeric_features = None
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "model.zip")
        if model_path:
            self.load(model_path)

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Initialize numeric_features based on the current game
        if self.numeric_features is None:
            num_players = len(game.state.players)
            self.features = get_feature_ordering(num_players)
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]

        # Generate observation from the game state
        obs = self.generate_observation(game)

        # Generate action mask from playable actions
        action_mask = self.generate_action_mask(playable_actions)

        # Predict the action index
        action_index, _ = self.model.predict(
            obs, action_masks=action_mask, deterministic=True
        )

        # Map the action index to the actual Action
        try:
            selected_action = self.action_index_to_action(
                action_index, playable_actions
            )
            return selected_action
        except Exception as e:
            print(f"Error mapping action index to Action: {e}")
            # Default to the first playable action
            return list(playable_actions)[0]

    def generate_observation(self, game: Game):
        # Create the sample
        sample = create_sample(game, self.color)

        # Generate board tensor
        board_tensor = create_board_tensor(
            game, self.color, channels_first=True
        ).astype(np.float32)

        # Extract numeric features
        numeric = np.array(
            [float(sample[i]) for i in self.numeric_features], dtype=np.float32
        )

        # Create the observation
        obs = {"board": board_tensor, "numeric": numeric}
        return obs

    def generate_action_mask(self, playable_actions: Iterable[Action]):
        action_mask = np.zeros(self.model.action_space.n, dtype=bool)
        for action in playable_actions:
            try:
                action_index = self.action_to_action_index(action)
                if (
                    action_index is not None
                    and 0 <= action_index < self.model.action_space.n
                ):
                    action_mask[action_index] = True
            except Exception as e:
                print(f"Error in action_to_action_index: {e}")
                continue
        return action_mask

    def action_to_action_index(self, action: Action):
        action_index = to_action_space(action)
        return action_index

    def action_index_to_action(
        self, action_index: int, playable_actions: Iterable[Action]
    ):
        action = from_action_space(action_index, playable_actions)
        if action in playable_actions:
            return action
        else:
            raise ValueError(f"Action {action} not in playable actions.")

    def load(self, path):
        self.model = MaskablePPO.load(
            path,
            custom_objects={
                "features_extractor_class": CustomCNN,
            },
        )
