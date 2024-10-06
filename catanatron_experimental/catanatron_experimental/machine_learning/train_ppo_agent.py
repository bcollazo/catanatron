import math
import os
os.environ["WANDB_DISABLE_SYMLINKS"] = "True"
from typing import Any
import atexit

import gymnasium as gym
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb
from wandb.integration.sb3 import WandbCallback

from catanatron import Color
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,
)
from catanatron.models.player import RandomPlayer
from catanatron.state_functions import get_actual_victory_points

LOAD = False

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float64)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])

def build_partial_rewards(vps_to_win):
    def partial_rewards(game, p0_color):
        winning_color = game.winning_color()
        if winning_color is None:
            return 0

        total = 0
        if p0_color == winning_color:
            total += 0.20
        else:
            total -= 0.20
        enemy_vps = [
            get_actual_victory_points(game.state, color)
            for color in game.state.colors
            if color != p0_color
        ]
        enemy_avg_vp = sum(enemy_vps) / len(enemy_vps)
        my_vps = get_actual_victory_points(game.state, p0_color)
        vp_diff = (my_vps - enemy_avg_vp) / (vps_to_win - 1)

        total += 0.80 * vp_diff
        print(f"my_vps = {my_vps} enemy_avg_vp = {enemy_avg_vp} partial_rewards = {total}")
        return total

    return partial_rewards

def learning_rate_schedule(initial_lr, final_lr):
    def lr_schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return lr_schedule

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN to process the board observations.
    :param observation_space: (gym.Space)
    :param cnn_arch: List of integers specifying the number of filters in each Conv layer.
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_arch,
        features_dim: int = 256,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space["board"].shape[0]

        layers = []
        in_channels = n_input_channels
        for out_channels in cnn_arch:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Compute the number of features after CNN
        with th.no_grad():
            sample_board = th.as_tensor(observation_space.sample()["board"][None]).float()
            n_flatten = self.cnn(sample_board).shape[1]

        n_numeric_features = observation_space["numeric"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_numeric_features, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        board_features = self.cnn(observations["board"])
        concatenated_tensor = th.cat([board_features, observations["numeric"]], dim=1)
        return self.linear(concatenated_tensor)

def main():
    # ===== Params:
    total_timesteps = 10_000
    cnn_arch = [64, 64, 32]
    net_arch = [dict(vf=[512, 256, 128], pi=[512, 256, 128])]
    activation_fn = th.nn.Tanh
    initial_lr = 1e-4
    final_lr = 1e-5
    ent_coef = 0.1
    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    enemies = [ValueFunctionPlayer(Color.RED)]
    reward_function = build_partial_rewards(vps_to_win)
    representation = "vector"
    batch_size = 8192
    gamma = 0.98
    normalized = False
    selfplay = False

    # Create learning rate schedule
    lr_schedule = learning_rate_schedule(initial_lr, final_lr)

    # Build Experiment Name
    iters = round(math.log(total_timesteps, 10))
    arch_str = (
        activation_fn.__name__
        + "x".join([str(i) for i in net_arch[:-1]])
        + "+"
        + "vf="
        + "x".join([str(i) for i in net_arch[-1]["vf"]])
        + "+"
        + "pi="
        + "x".join([str(i) for i in net_arch[-1]["pi"]])
    )
    if representation == "mixed":
        arch_str = "Cnn" + "x".join([str(i) for i in cnn_arch]) + "+" + arch_str
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = f"ppo-{selfplay}-{normalized}-{iters}-{batch_size}-{gamma}-{enemy_desc}-{reward_function.__name__}-{representation}-{arch_str}-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
    print(experiment_name)

    # WandB config
    config = {
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "total_timesteps": total_timesteps,
        "net_arch": net_arch,
        "activation_fn": activation_fn.__name__,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "enemies": [str(enemy) for enemy in enemies],
        "reward_function": reward_function.__name__,
        "representation": representation,
        "batch_size": batch_size,
        "gamma": gamma,
        "normalized": normalized,
        "cnn_arch": cnn_arch,
        "selfplay": selfplay,
        "experiment_name": experiment_name,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    # Init Environment and Model
    env = gym.make(
        env_name,
        config={
            "map_type": map_type,
            "vps_to_win": vps_to_win,
            "enemies": enemies,
            "reward_function": reward_function,
            "representation": representation,
            "normalized": True,
        },
    )
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Print the observation space to verify its type
    print("Observation Space:", env.observation_space)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    path = "catanatron_experimental/catanatron_experimental/machine_learning/model.zip"
    try:
        model = MaskablePPO.load(path, env, device=device)
        print("Loaded", "model.zip")
    except Exception as e:
        print(f"Failed to load the model from {path}: {e}")
        print("Creating a new model.")
        policy_kwargs: Any = dict(activation_fn=activation_fn, net_arch=net_arch[0])
        if representation == "mixed":
            policy_kwargs["features_extractor_class"] = CustomCNN
            policy_kwargs["features_extractor_kwargs"] = dict(
                cnn_arch=cnn_arch,
                features_dim=512  # Adjust as needed
            )
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            gamma=gamma,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log="./logs/mppo_tensorboard/" + experiment_name,
            device=device
        )

    # Save a checkpoint every 100,000 steps
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path="./logs/", name_prefix=experiment_name
    )
    callback = CallbackList([checkpoint_callback])#, wandb_callback])

    if selfplay:
        selfplay_iterations = 10
        for i in range(selfplay_iterations):
            model.learn(
                total_timesteps=int(total_timesteps / selfplay_iterations),
                callback=callback,
            )
            model.save(path)
    else:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(path)

    model.save(path)
    run.finish()

if __name__ == "__main__":
    main()
