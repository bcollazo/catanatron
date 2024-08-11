import os
import argparse
from typing import Iterable

import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer, Player
from catanatron.models.actions import Action
from catanatron_gym.envs.catanatron_env import CatanatronEnv

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
        dummy_game = Game([self, RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE), RandomPlayer(Color.ORANGE)])
        env = self.create_env(dummy_game)
        env = ActionMasker(env, self.mask_fn)
        self.model = MaskablePPO.load(path, env=env)

class GameCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(GameCallback, self).__init__(verbose)
        self.game_rewards = []

    def _on_step(self):
        return True

    def on_rollout_end(self):
        try:
            # Attempt to access game_reward if it's available
            game_reward = self.training_env.get_attr('game_reward')[0]
        except AttributeError:
            # Fallback: Use the cumulative reward if game_reward isn't defined
            game_reward = sum(self.locals['rewards'])
        
        self.game_rewards.append(game_reward)
        self.logger.record('game_reward', game_reward)


def train_model(episodes, model_path=None, learning_rate=3e-4, eval_freq=10):
    player = MyPlayer(Color.RED, model_path)
    
    if model_path and os.path.exists(model_path):
        player.model = MaskablePPO.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        dummy_game = Game([player, RandomPlayer(Color.BLUE)])
        player.initialize_model(dummy_game)
        print("Initialized new model")

    lr_schedule = get_linear_fn(learning_rate, learning_rate * 0.1, episodes)

    callback = GameCallback()

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        
        if episode % eval_freq == 0:
            eval_reward = evaluate_model(player.model)
            print(f"Evaluation reward: {eval_reward}")

        game = Game([player, RandomPlayer(Color.BLUE)])
        env = CatanatronEnv(config={
            "map_type": game.state.board.map.__class__.__name__,
            "discard_limit": game.state.discard_limit,
            "vps_to_win": game.vps_to_win,
            "num_players": len(game.state.colors),
            "player_colors": game.state.colors,
            "catan_map": game.state.board.map,
        })
        env = ActionMasker(env, player.mask_fn)
        player.model.set_env(env)
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        for _ in range(1000):  # Set a maximum number of steps per episode
            action, _ = player.model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            if done:
                break

        print(f"Episode finished. Reward: {episode_reward}")

        # Update the learning rate
        current_lr = lr_schedule(episode / episodes)
        player.model.learning_rate = current_lr

        player.model.learn(total_timesteps=1, callback=callback, reset_num_timesteps=False)

        if episode % 100 == 0:
            player.model.save(f"model_checkpoint_{episode}.zip")

    save_path = model_path or "trained_model.zip"
    player.model.save(save_path)
    print(f"Model saved to {save_path}.")
    
    return callback.game_rewards

def evaluate_model(model, num_games=10):
    total_reward = 0
    for game_num in range(num_games):
        print(f"Evaluation game {game_num + 1}/{num_games}")
        game = Game([MyPlayer(Color.RED), RandomPlayer(Color.BLUE)])
        env = CatanatronEnv(config={
            "map_type": game.state.board.map.__class__.__name__,
            "discard_limit": game.state.discard_limit,
            "vps_to_win": game.vps_to_win,
            "num_players": len(game.state.colors),
            "player_colors": game.state.colors,
            "catan_map": game.state.board.map,
        })
        env = ActionMasker(env, MyPlayer(Color.RED).mask_fn)
        
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        done = False
        episode_reward = 0
        step_count = 0
        while not done and step_count < 1000:
            print(f"Step {step_count}")
            action_mask = env.action_masks()
            print(f"Action mask shape: {action_mask.shape}, sum: {action_mask.sum()}")
            try:
                action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                print(f"Predicted action: {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                print(f"Reward: {reward}, Done: {done}")
            except Exception as e:
                print(f"Error during prediction: {e}")
                break
            step_count += 1
        
        total_reward += episode_reward
        print(f"Game {game_num + 1} finished. Reward: {episode_reward}")
    
    avg_reward = total_reward / num_games
    print(f"Average evaluation reward: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Catan AI model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--model", type=str, help="Path to load/save the model")
    args = parser.parse_args()

    train_model(args.episodes, args.model)