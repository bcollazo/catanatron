from typing import Iterable
import numpy as np
from sb3_contrib import MaskablePPO

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import from_action_space, to_action_space
from catanatron_gym.features import create_sample, get_feature_ordering

class PPOPlayer(Player):
    """
    Proximal Policy Optimization (PPO) reinforcement learning agent.
    """

    def __init__(self, color, model_path=None):
        super().__init__(color)
        self.model = None
        if model_path is None:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, '..', 'model.zip')
        if model_path:
            self.load(model_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Generate observation from the game state
        obs = self.generate_observation(game)

        # Generate action mask from playable actions
        action_mask = self.generate_action_mask(playable_actions)

        # Predict the action index
        action_index, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)

        # Map the action index to the actual Action
        try:
            selected_action = self.action_index_to_action(action_index, playable_actions)
            return selected_action
        except Exception as e:
            print(f"Error mapping action index to Action: {e}")
            # Default to the first playable action
            return list(playable_actions)[0]

    def generate_observation(self, game: Game):
        # Create the observation using create_sample
        sample = create_sample(game, self.color)
        # Get the list of features used
        features = get_feature_ordering(num_players=len(game.state.players))
        # Create the observation vector
        obs = np.array([float(sample[i]) for i in features], dtype=np.float32)
        return obs

    def generate_action_mask(self, playable_actions: Iterable[Action]):
        action_mask = np.zeros(self.model.action_space.n, dtype=bool)
        for action in playable_actions:
            try:
                action_index = self.action_to_action_index(action)
                if action_index is not None and 0 <= action_index < self.model.action_space.n:
                    action_mask[action_index] = True
            except Exception as e:
                print(f"Error in action_to_action_index: {e}")
                continue
        return action_mask

    def action_to_action_index(self, action: Action):
        action_index = to_action_space(action)
        return action_index

    def action_index_to_action(self, action_index: int, playable_actions: Iterable[Action]):
        action = from_action_space(action_index, playable_actions)
        if action in playable_actions:
            return action
        else:
            raise ValueError(f"Action {action} not in playable actions.")

    def load(self, path):
        self.model = MaskablePPO.load(path)
