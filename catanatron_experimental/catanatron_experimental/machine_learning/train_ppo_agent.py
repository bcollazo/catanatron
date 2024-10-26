import math
import os
import random
os.environ["WANDB_DISABLE_SYMLINKS"] = "True"
from typing import Any
import atexit
import time
import multiprocessing

from functools import partial
import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb
from wandb.integration.sb3 import WandbCallback

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,
)
from catanatron.models.player import RandomPlayer

# Import shared functions from the separate module
from reward_functions import partial_rewards, mask_fn

LOAD = False

def learning_rate_schedule(initial_lr, final_lr):
    def lr_schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return lr_schedule

def main():
    # Remove the set_start_method from here

    # ===== Params:
    # With 100,000,000 timesteps, training took 4.97 days
    total_timesteps = 100_000_000
    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(
        vf=[4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256],
        pi=[4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256]
    )]
    activation_fn = th.nn.LeakyReLU
    initial_lr = 1e-4
    final_lr = 1e-6
    ent_coef = 0.01
    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    enemies = [ValueFunctionPlayer(Color.RED)]
    reward_function = partial(partial_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = partial_rewards.__name__
    representation = "mixed"
    # batch_size = 64
    gamma = 0.99
    normalized = False
    selfplay = False
    seed = 42

    n_envs = 8
    n_steps = 256
    batch_size = n_envs * n_steps
    n_epochs = 10

    assert (n_envs * n_steps) % batch_size == 0, "batch_size must divide n_envs * n_steps"

    start_time = time.time()

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

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
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "seed": seed,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    # Define the environment creation function
    def make_env(rank, seed=0):
        def _init():
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
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    # Print the observation space to verify its type
    print("Observation Space:", env.observation_space)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model')  # Removed '.zip'
    try:
        model = MaskablePPO.load(model_path, env, device=device)
        # Override the training configuration from the previously trained model
        model.gamma = gamma
        model.ent_coef = ent_coef
        model.learning_rate = lr_schedule
        model.batch_size = batch_size
        model._setup_lr_schedule()
        print("Loaded", "model")
    except Exception as e:
        print(f"Failed to load the model from {model_path}: {e}")
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
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log="./logs/mppo_tensorboard/" + experiment_name,
            device=device,
            seed=seed
        )

    # Save a checkpoint every 100,000 steps
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path="./logs/", name_prefix=experiment_name
    )
    callback = CallbackList([checkpoint_callback])  # , wandb_callback])

    if selfplay:
        selfplay_iterations = 10
        for i in range(selfplay_iterations):
            model.learn(
                total_timesteps=int(total_timesteps / selfplay_iterations),
                callback=callback,
            )
            model.save(model_path)
    else:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(model_path)

        elapsed_time = time.time() - start_time
        timesteps_per_second = total_timesteps / elapsed_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")
        print(f"Timesteps per second = {timesteps_per_second:.2f}")

    model.save(model_path)
    run.finish()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
