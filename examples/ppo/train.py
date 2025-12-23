"""
Restartable Stable Baselines3 training example with Weights & Biases logging.

Features:
- Vectorized environments (parallel games for speedup)
- VecNormalize (observation and reward normalization)
- Shaped reward function (incremental rewards for progress)
- Linear learning rate schedule (decreases over time)
- GPU support (automatic detection)
- Checkpoint saving/loading for resumable training
- Weights & Biases integration for monitoring

Usage:
    # Start new training:
    python train.py

    # Resume from checkpoint:
    python train.py --resume

    # Run a wandb sweep:
    python train.py --sweep

    # Train for custom timesteps:
    python train.py --timesteps 500000
"""

import random
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

import catanatron.gym
from ppo_utils import make_catan_env


# Configuration object (compatible with wandb)
DEFAULT_CONFIG = {
    # Environment parameters
    "map_type": "MINI",  # Map type for Catan (BASE, MINI, etc.)
    "vps_to_win": 6,  # Victory points needed to win
    "use_shaped_reward": True,  # Use shaped vs simple reward function
    # PPO hyperparameters
    "n_envs": 4,  # Number of parallel environments
    "n_steps": 1024,  # Number of steps to collect before update
    "batch_size": 128,  # Batch size for training
    "n_epochs": 10,  # Number of epochs for PPO update
    "gamma": 0.99,  # Discount factor for future rewards
    "initial_lr": 0.01,  # Initial learning rate
    "lr_decay_orders": 1,  # Orders of magnitude to decay to (final = initial / 10^orders)
    "ent_coef": 0.01,  # Entropy coefficient for exploration
    # Network architecture
    "num_layers": 3,  # Number of hidden layers
    "neurons_per_layer": 256,  # Neurons in each layer
    # Training parameters
    "seed": 42,  # Random seed
    "checkpoint_freq": 10_000,  # Save checkpoint every N steps
}

# Wandb sweep configuration
SWEEP_CONFIG = {
    "method": "random",
    "metric": {"name": "rollout/ep_rew_mean", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256, 512]},
        "n_steps": {"values": [512, 1024, 2048, 4096]},
        "n_epochs": {"values": [5, 10, 15, 20]},
        "gamma": {"values": [0.9, 0.95, 0.99, 0.999]},
        "initial_lr": {"values": [1.0, 0.1, 0.01, 0.001, 0.0001]},
        "lr_decay_orders": {"values": [1, 2, 3, 4]},
        "ent_coef": {"values": [0.0, 0.005, 0.01, 0.02]},
        "num_layers": {"values": [1, 3, 5, 10]},
        "neurons_per_layer": {"values": [128, 256, 512, 1024]},
    },
}

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train_model(run, args, cfg):
    """Execute the training loop."""
    # Validate configuration
    assert (cfg["n_envs"] * cfg["n_steps"]) % cfg["batch_size"] == 0, (
        "BATCH_SIZE must divide N_ENVS * N_STEPS"
    )

    # Set random seeds for reproducibility
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device (GPU if available)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Create vectorized environments
    print(
        f"Using reward function: {'shaped (incremental)' if cfg['use_shaped_reward'] else 'simple (sparse)'}"
    )
    print(f"Using map type: {cfg['map_type']}, VPs to win: {cfg['vps_to_win']}")
    print(f"Creating {cfg['n_envs']} parallel environments...")
    env = make_vec_env(
        lambda: make_catan_env(cfg),
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
        vec_env_cls=SubprocVecEnv,  # Use subprocesses for CPU-heavy environments
    )

    # Wrap with VecNormalize for observation and reward normalization
    print("Wrapping environments with VecNormalize (obs + reward normalization)")
    env = VecNormalize(
        env,
        norm_obs=True,  # Normalize observations
        norm_reward=False,  # Normalize rewards
        clip_obs=10.0,  # Clip observations to [-10, 10] after normalization
        clip_reward=10.0,  # Clip rewards to [-10, 10] after normalization
    )

    # Load or create model
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if args.resume:
        checkpoint = get_latest_checkpoint()
        if checkpoint:
            print(f"Loading checkpoint: {checkpoint}")

            # Load VecNormalize stats if available
            vec_normalize_path = checkpoint.replace(".zip", "_vecnormalize.pkl")
            if os.path.exists(vec_normalize_path):
                print(f"Loading VecNormalize stats from: {vec_normalize_path}")
                env = VecNormalize.load(vec_normalize_path, env)

            model = MaskablePPO.load(
                checkpoint,
                env=env,
                device=device,
                tensorboard_log=f"runs/{run.id}",
            )
        else:
            print("No checkpoint found, starting fresh")
            args.resume = False

    if not args.resume:
        # Configure network architecture
        net_arch = [cfg["neurons_per_layer"]] * cfg["num_layers"]
        policy_kwargs = dict(net_arch=net_arch)

        # Create learning rate schedule
        final_lr = compute_final_lr(cfg["initial_lr"], cfg["lr_decay_orders"])
        lr_schedule = linear_schedule(cfg["initial_lr"], final_lr)

        print(f"Creating new model with architecture: {net_arch}")
        print(
            f"PPO config: n_steps={cfg['n_steps']}, batch_size={cfg['batch_size']}, n_epochs={cfg['n_epochs']}, gamma={cfg['gamma']}, ent_coef={cfg['ent_coef']}"
        )
        print(f"Learning rate schedule: {cfg['initial_lr']:.2e} → {final_lr:.2e}")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            learning_rate=lr_schedule,
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            gamma=cfg["gamma"],
            n_epochs=cfg["n_epochs"],
            ent_coef=cfg["ent_coef"],
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            seed=cfg["seed"],
            policy_kwargs=policy_kwargs,
            device=device,
        )

    # Setup checkpoint callback (also saves VecNormalize stats)
    checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=cfg["checkpoint_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
    )

    # Setup wandb callback
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    # Train
    print(f"\nTraining for {args.timesteps:,} timesteps")
    print(f"With {cfg['n_envs']} parallel environments (~{cfg['n_envs']}x speedup)")
    print(f"Wandb run: {run.get_url()}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, wandb_callback],
        reset_num_timesteps=not args.resume,
    )

    # Save final model and VecNormalize stats
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.zip")
    model.save(final_path)

    vec_normalize_path = os.path.join(CHECKPOINT_DIR, "final_model_vecnormalize.pkl")
    env.save(vec_normalize_path)

    print(f"\nDone! Final model: {final_path}")
    print(f"VecNormalize stats: {vec_normalize_path}")

    # Clean up
    env.close()


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a wandb sweep over the sweep configuration",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help=f"Number of timesteps to train for (default: {100_000:,})",
    )
    args = parser.parse_args()

    # Login to wandb
    wandb.login()

    def run_training():
        with wandb.init(
            project="catan-ppo",
            config=DEFAULT_CONFIG,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # save the code
        ) as run:
            train_model(run, args, dict(wandb.config))

    if args.sweep:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="catan-ppo")
        wandb.agent(sweep_id, function=run_training)
    else:
        run_training()


def get_latest_checkpoint():
    """Find the most recent checkpoint."""
    checkpoint_path = Path(CHECKPOINT_DIR)
    if not checkpoint_path.exists():
        return None

    checkpoints = list(checkpoint_path.glob("rl_model_*_steps.zip"))
    if not checkpoints:
        return None

    # Get latest by timestep number
    latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[2]))
    return str(latest)


def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def compute_final_lr(initial_lr, lr_decay_orders):
    return initial_lr / (10**lr_decay_orders)


class VecNormalizeCheckpointCallback(CheckpointCallback):
    """Custom checkpoint callback that also saves VecNormalize statistics."""

    def _on_step(self) -> bool:
        # Save model checkpoint (parent class behavior)
        result = super()._on_step()

        # Also save VecNormalize stats if the model was just saved
        if result and isinstance(self.model.get_env(), VecNormalize):
            # Get the path of the most recently saved model
            checkpoint_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            vec_normalize_path = checkpoint_path.replace(".zip", "_vecnormalize.pkl")

            # Save VecNormalize statistics
            self.model.get_env().save(vec_normalize_path)

        return result


if __name__ == "__main__":
    main()
