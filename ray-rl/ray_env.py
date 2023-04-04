import os
from typing import Iterable
from pprint import pprint

from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron import Game, RandomPlayer, Color
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron_gym.envs.catanatron_env import (
    to_action_space,
    CatanatronEnv,
    from_action_space,
)
from catanatron_gym.features import create_sample, get_feature_ordering
from catanatron_experimental.play import play_batch
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer

from ray_model import register_model


base_path = os.path.expanduser("~/BryanCode/catanatron/logs")
experiment_name = "PPO_selflearn_64xx6relu-3"

# ===== ENVIRONMENT
class RayCatanatronEnv(gym.Env):
    def __init__(self, env_config):
        self.inner_env = CatanatronEnv(env_config)
        self.resets = 0
        print("BUILT ENVIRONMENT", self.inner_env, env_config)
        self.action_space = self.inner_env.action_space
        obs_space = spaces.Box(
            low=0,
            high=self.inner_env.observation_space.high,
            shape=self.inner_env.observation_space.shape,
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                # "valid_actions": spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int16),
                "actual_obs": obs_space,
                "action_mask": spaces.Box(
                    0, 1, shape=(self.action_space.n,), dtype=np.float32
                ),
            }
        )
        # self.spec.max_episode_steps = 1000 * 4 * 2
        self.enemy = RandomPlayer(Color.RED)

    def reset(self, *, seed=None, options=None):
        self.resets += 1

        # Set ENEMY depending on count of resets
        if self.resets % 100 == 0 and self.resets != 0:
            print("RESETTING ENEMYS")
            self.enemy = get_old_rayplayer(base_path, experiment_name, Color.RED)
        # enemy_list = [
        #     RandomPlayer(Color.RED),
        #     # WeightedRandomPlayer(Color.RED),
        #     # VictoryPointPlayer(Color.RED),
        #     # ValueFunctionPlayer(Color.RED),
        # ]
        # enemy_index = min(self.resets // 200, len(enemy_list) - 1)
        # enemy = enemy_list[enemy_index]

        # Set ENEMY
        self.inner_env.enemies = [self.enemy]
        self.inner_env.players = [self.inner_env.p0, self.enemy]
        print("RESET", self.resets, self.enemy)

        # get observation and mask
        observation, info = self.inner_env.reset(seed=seed, options=options)
        valid_actions = self.inner_env.get_valid_actions()
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        mask[valid_actions] = 1

        return {"actual_obs": np.array(observation), "action_mask": mask}, info

    def step(self, action):
        observation, reward, done, info = self.inner_env.step(action)
        valid_actions = self.inner_env.get_valid_actions()
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        mask[valid_actions] = 1

        return (
            {"actual_obs": observation, "action_mask": mask},
            reward,
            done,
            False,
            info,
        )


def env_creator(env_config):
    return RayCatanatronEnv(env_config)  # return an env instance


def register_catanatron_env():
    register_env("ray_catanatron_env", env_creator)


def get_old_rayplayer(base_path, experiment_name, color):
    # Take all checkpoints... if cant beat basic bots, use them
    checkpoints = [
        path
        for path in os.listdir(os.path.join(base_path, experiment_name))
        if path.startswith("checkpoint")
    ]
    assert (
        len(checkpoints) > 3
    )  # every 15 iterations (ea iteration are like 20 games per worker, so ~80 games)
    old_checkpoint_filename = sorted(checkpoints)[-3]
    old_checkpoint_path = os.path.join(
        base_path, experiment_name, old_checkpoint_filename
    )

    print("LOADING PREVIOUSLY SAVED CHECKPOINT", old_checkpoint_path)
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
        .environment(env="ray_catanatron_env")
        .training(
            lr=1e-4,
            model={
                "custom_model": "action_mask_model",
                "custom_model_config": {
                    "fcnet_hiddens": [64, 64, 64, 64, 64, 64],
                    "fcnet_activation": "relu",
                },
            },
        )
        .build()
    )
    algo.restore(old_checkpoint_path)
    # algo = Algorithm.from_checkpoint(old_checkpoint_path)

    class RayPlayer(Player):
        def __init__(self, checkpoint_path, color, is_bot=True):
            super().__init__(color, is_bot)
            self.checkpoint_path = checkpoint_path

        def __repr__(self):
            return f"{type(self).__name__}:({self.checkpoint_path}){self.color.value}"

        def decide(self, game: Game, playable_actions: Iterable[Action]):
            """Should return one of the playable_actions.

            Args:
                game (Game): complete game state. read-only.
                playable_actions (Iterable[Action]): options to choose from
            Return:
                action (Action): Chosen element of playable_actions
            """
            # ===== YOUR CODE HERE =====
            # TODO: Use CatanatronEnv?
            features = get_feature_ordering(len(game.state.colors), "BASE")
            sample = create_sample(game, self.color)  # assuming vec rep
            actual_obs = np.array([float(sample[i]) for i in features])

            valid_actions = list(map(to_action_space, playable_actions))
            mask = np.zeros(CatanatronEnv.action_space.n, dtype=np.float32)
            mask[valid_actions] = 1

            obs = {"actual_obs": np.array(actual_obs), "action_mask": mask}
            result = algo.compute_single_action(obs)
            action = from_action_space(int(result), playable_actions)

            return action
            # ===== END YOUR CODE =====

    return RayPlayer(os.path.join(experiment_name, old_checkpoint_filename), color)


if __name__ == "__main__":
    register_catanatron_env()
    register_model()
    # Play a simple 4v4 game
    players = [
        # RandomPlayer(Color.RED),
        # WeightedRandomPlayer(Color.RED),
        # VictoryPointPlayer(Color.RED),  # 80/20
        ValueFunctionPlayer(Color.RED),  # 0/100
        get_old_rayplayer(base_path, experiment_name, Color.ORANGE),
    ]
    wins, results_by_player, games = play_batch(50, players)
