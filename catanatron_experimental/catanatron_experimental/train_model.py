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
from catanatron_gym.envs.catanatron_env import CatanatronEnv, to_action_space

from my_player import MyPlayer
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
            # print(f"Evaluation reward: {eval_reward}")

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
        while not done:
            action, _ = player.model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode finished. Total Reward: {episode_reward}")

        # Update the learning rate
        current_lr = lr_schedule(episode / episodes)
        player.model.learning_rate = current_lr

        player.model.learn(total_timesteps=100, callback=callback, reset_num_timesteps=False)

        if episode % 100 == 0 and episode != 0:
            player.model.save(f"model_checkpoint_{episode}.zip")

    save_path = model_path or "trained_model.zip"
    player.model.save(save_path)
    print(f"Model saved to {save_path}.")
    
    return callback.game_rewards

def evaluate_model(model, num_games=10):
    print("Starting model evaluation...")
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
        while not done and step_count < 10000:
            # print(f"Step {step_count}")
            action_mask = env.action_masks()
            # print(f"Action mask shape: {action_mask.shape}, sum: {action_mask.sum()}")
            try:
                action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                # print(f"Predicted action: {action}")
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