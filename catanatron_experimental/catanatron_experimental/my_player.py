from typing import Iterable
import os
import zipfile

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import get_linear_fn

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player, Color, RandomPlayer
from catanatron_gym.envs.catanatron_env import CatanatronEnv, from_action_space, to_action_space
from catanatron_experimental.cli.cli_players import register_player

@register_player("MP")
class MyPlayer(Player):
    def __init__(self, color, model_path=None):
        super().__init__(color)
        self.model = None
        self.env = None
        self.learning_rate_schedule = get_linear_fn(3e-4, 3e-5, 1.0)
        self.clip_range_schedule = get_linear_fn(0.2, 0.02, 1.0)
        if model_path:
            self.load(model_path)

    def __getstate__(self):
        state = self.__dict__.copy()
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
            obs, _ = self.env.reset()
        else:
            obs = self.env._get_observation()

        action_mask = self.env.get_action_mask(playable_actions)

        # Predict the action index based on the full action space
        action_index, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        # print(f"Predicted action index: {action_index}, Playable actions length: {len(playable_actions)}")

        try:
            # Convert the action index to an actual Action object using from_action_space
            selected_action = from_action_space(action_index, playable_actions)
            return selected_action
        except ValueError as e:
            print(f"Error: {e}. Defaulting to the first playable action. {playable_actions[0]}")
            return list(playable_actions)[0]

    def initialize_model(self, game):
        self.env = self.create_env(game)
        env = ActionMasker(self.env, self.action_mask_fn)

        self.model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate_schedule,
            clip_range=self.clip_range_schedule,
            verbose=1
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

    def action_mask_fn(self, env):
        return env.action_masks

    @property
    def mask_fn(self):
        return self.action_mask_fn

    def train(self, total_timesteps=10000):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        self.model.save(path)

    def load(self, path):
        dummy_game = Game([self, RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE), RandomPlayer(Color.ORANGE)])
        env = self.create_env(dummy_game)
        env = ActionMasker(env, self.action_mask_fn)
        self.model = MaskablePPO.load(path, env=env)
