"""
Restartable Stable Baselines3 training example with TensorBoard logging.

Features:
- Vectorized environments (parallel games for speedup)
- VecNormalize (observation and reward normalization)
- Shaped reward function (incremental rewards for progress)
- Linear learning rate schedule (decreases over time)
- GPU support (automatic detection)
- Checkpoint saving/loading for resumable training
- TensorBoard integration for monitoring
- Wandb-ready configuration object

Usage:
    # Start new training:
    python train.py

    # Resume from checkpoint:
    python train.py --resume

    # Train for custom timesteps:
    python train.py --timesteps 500000

    # View TensorBoard:
    tensorboard --logdir ./tensorboard_logs

    # To use with wandb (add to main()):
    import wandb
    wandb.init(project="catan-ppo", config=config)
"""

import random
import argparse
import os
from pathlib import Path

import gymnasium
import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import catanatron.gym
from catanatron import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import simple_reward
from shaped_reward import ShapedRewardFunction


# Configuration object (compatible with wandb)
config = {
    # Environment parameters
    "map_type": "MINI",  # Map type for Catan (BASE, MINI, etc.)
    "vps_to_win": 6,  # Victory points needed to win
    "use_shaped_reward": True,  # Use shaped vs simple reward function
    # PPO hyperparameters
    "n_envs": 32,  # Number of parallel environments
    "n_steps": 2048,  # Number of steps to collect before update
    "batch_size": 256,  # Batch size for training
    "gamma": 0.99,  # Discount factor for future rewards
    "initial_lr": 0.01,  # Initial learning rate
    "final_lr": 0.001,  # Final learning rate
    "ent_coef": 0.01,  # Entropy coefficient for exploration
    # Network architecture
    "num_layers": 3,  # Number of hidden layers
    "neurons_per_layer": 256,  # Neurons in each layer
    # Training parameters
    "seed": 42,  # Random seed
    "checkpoint_freq": 10_000,  # Save checkpoint every N steps
}

# Validate configuration
assert (config["n_envs"] * config["n_steps"]) % config["batch_size"] == 0, (
    "BATCH_SIZE must divide N_ENVS * N_STEPS"
)

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TENSORBOARD_DIR = os.path.join(os.path.dirname(__file__), "tensorboard_logs")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help=f"Number of timesteps to train for (default: {100_000:,})",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
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
        f"Using reward function: {'shaped (incremental)' if config['use_shaped_reward'] else 'simple (sparse)'}"
    )
    print(f"Using map type: {config['map_type']}, VPs to win: {config['vps_to_win']}")
    print(f"Creating {config['n_envs']} parallel environments...")
    env = make_vec_env(
        make_catan_env,
        n_envs=config["n_envs"],
        seed=config["seed"],
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
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
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
                tensorboard_log=TENSORBOARD_DIR,
                device=device,
            )
        else:
            print("No checkpoint found, starting fresh")
            args.resume = False

    if not args.resume:
        # Configure network architecture
        net_arch = [config["neurons_per_layer"]] * config["num_layers"]
        policy_kwargs = dict(net_arch=net_arch)

        # Create learning rate schedule
        lr_schedule = linear_schedule(config["initial_lr"], config["final_lr"])

        print(f"Creating new model with architecture: {net_arch}")
        print(
            f"PPO config: n_steps={config['n_steps']}, batch_size={config['batch_size']}, gamma={config['gamma']}, ent_coef={config['ent_coef']}"
        )
        print(
            f"Learning rate schedule: {config['initial_lr']:.2e} → {config['final_lr']:.2e}"
        )
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            learning_rate=lr_schedule,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            ent_coef=config["ent_coef"],
            verbose=1,
            tensorboard_log=TENSORBOARD_DIR,
            seed=config["seed"],
            policy_kwargs=policy_kwargs,
            device=device,
        )

    # Setup checkpoint callback (also saves VecNormalize stats)
    checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
    )

    # Train
    print(f"\nTraining for {args.timesteps:,} timesteps")
    print(
        f"With {config['n_envs']} parallel environments (~{config['n_envs']}x speedup)"
    )
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}\n")

    # Create experiment name with parameters
    reward_type = "shaped" if config["use_shaped_reward"] else "simple"
    experiment_name = (
        f"MaskablePPO_{config['map_type']}_vp{config['vps_to_win']}_envs{config['n_envs']}_steps{config['n_steps']}_batch{config['batch_size']}_"
        f"gamma{config['gamma']}_lr{config['initial_lr']}-{config['final_lr']}_ent{config['ent_coef']}_"
        f"layers{config['num_layers']}x{config['neurons_per_layer']}_{reward_type}"
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not args.resume,
        tb_log_name=experiment_name,
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


def make_catan_env():
    """Factory function to create a Catan environment for vectorization."""
    # Create fresh reward function instance for each environment
    reward_fn = ShapedRewardFunction() if config["use_shaped_reward"] else simple_reward

    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": config["map_type"],
            "vps_to_win": config["vps_to_win"],
            "enemies": [ValueFunctionPlayer(Color.RED)],
            "reward_function": reward_fn,
        },
    )
    env = ActionMasker(env, mask_fn)
    return env


def mask_fn(env) -> np.ndarray:
    """Create action mask for valid actions."""
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])


if __name__ == "__main__":
    main()
