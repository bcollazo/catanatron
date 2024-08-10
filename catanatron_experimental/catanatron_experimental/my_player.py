from typing import Iterable
from catanatron_experimental.cli.cli_players import register_player
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import CatanatronEnv
import os
import zipfile

print("Defining MyPlayer...")

class LinearSchedule:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

@register_player("MP")
class MyPlayer(Player):
    def __init__(self, color, model_path=None):
        super().__init__(color)
        self.model = None
        self.env = None
        self.learning_rate_schedule = LinearSchedule(3e-4)
        self.clip_range_schedule = LinearSchedule(0.2)
        if model_path:
            self.load(model_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the model and env from the state as they are not pickleable
        state['model'] = None
        state['env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None
        self.env = None

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if self.model is None:
            self.initialize_model(game)

        if self.env is None or self.env.game is not game:
            self.env = self.create_env(game)
            obs, _ = self.env.reset()  # Get initial observation
        else:
            obs, _ = self.env.reset()

        action_mask = self.get_action_mask(playable_actions, self.env.action_space.n)

        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return list(playable_actions)[action]

    def initialize_model(self, game):
        self.env = self.create_env(game)
        env = ActionMasker(self.env, self.mask_fn)

        self.model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate_schedule,
            clip_range=self.clip_range_schedule,
            verbose=1,
            device='cuda'  # Enable CUDA
        )

    def create_env(self, game):
        return CatanatronEnv(config={
            "map_type": game.state.board.map.__class__.__name__,
            "discard_limit": game.state.discard_limit,
            "vps_to_win": game.vps_to_win,
            "num_players": len(game.state.colors),
            "player_colors": game.state.colors,
            "catan_map": game.state.board.map,
        })

    def get_action_mask(self, playable_actions, action_space_n):
        mask = np.zeros(action_space_n, dtype=bool)
        for i, action in enumerate(playable_actions):
            mask[i] = True
        return mask

    def mask_fn(self, env):
        return self.get_action_mask(env.get_valid_actions(), env.action_space.n)

    def train(self, total_timesteps=10000):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        self.model.save(path)

    def load(self, path):
        if not path.endswith(".zip"):
            path += ".zip"

        if not os.path.exists(path) or not zipfile.is_zipfile(path):
            # Initialize and save a model if it does not exist or is not valid
            print(f"Creating a new model and saving to {path}")
            dummy_game = Game([self, Player(None), Player(None), Player(None)])
            self.initialize_model(dummy_game)
            self.train(1000)  # Train for a small number of steps to create a valid model
            self.save(path)
            print(f"Created and saved new model to {path}")

        if not zipfile.is_zipfile(path):
            raise ValueError(f"Error: the file {path} wasn't a zip-file")

        dummy_game = Game([self, Player(None), Player(None), Player(None)])
        env = self.create_env(dummy_game)
        env = ActionMasker(env, self.mask_fn)
        print(f"Loading model from {path}")
        self.model = MaskablePPO.load(path, env=env)
        print(f"Loaded model from {path}")

print(f"MyPlayer defined as: {MyPlayer}")
