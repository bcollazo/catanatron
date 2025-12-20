"""
Restartable Stable Baselines3 training example with TensorBoard logging.

Features:
- Vectorized environments (8 parallel games for ~8x speedup)
- Shaped reward function (incremental rewards for progress)
- Linear learning rate schedule (decreases over time)
- GPU support (automatic detection)
- Checkpoint saving/loading for resumable training
- TensorBoard integration for monitoring

Configure by editing constants:
    N_ENVS = 8                      # Parallel environments
    NUM_LAYERS = 3                  # Number of hidden layers
    NEURONS_PER_LAYER = 256         # Neurons in each layer
    N_STEPS = 2048                  # PPO rollout steps
    BATCH_SIZE = 256                # PPO batch size
    INITIAL_LR = 0.0003             # Initial learning rate
    FINAL_LR = 0.00001              # Final learning rate
    ENT_COEF = 0.05                 # Entropy coefficient for exploration
    USE_SHAPED_REWARD = True        # Incremental vs sparse rewards

Usage:
    # Start new training:
    python train.py

    # Resume from checkpoint:
    python train.py --resume

    # Train for custom timesteps:
    python train.py --timesteps 500000

    # View TensorBoard:
    tensorboard --logdir ./tensorboard_logs
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
from stable_baselines3.common.vec_env import SubprocVecEnv

import catanatron.gym
from catanatron import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import simple_reward
from shaped_reward import ShapedRewardFunction


# Configuration
SEED = 42
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TENSORBOARD_DIR = os.path.join(os.path.dirname(__file__), "tensorboard_logs")
CHECKPOINT_FREQ = 10_000
NUM_LAYERS = 3
NEURONS_PER_LAYER = 256
USE_SHAPED_REWARD = True

# PPO parameters
N_ENVS = 8  # Number of parallel environments
N_STEPS = 2048  # Number of steps to collect before update (default: 2048)
BATCH_SIZE = 256  # Batch size for training (default: 64)
assert (N_ENVS * N_STEPS) % BATCH_SIZE == 0, "BATCH_SIZE must divide N_ENVS * N_STEPS"
# Learning rate schedule
INITIAL_LR = 0.0003  # Initial learning rate
FINAL_LR = 0.00001  # Final learning rate
ENT_COEF = 0.05  # Entropy coefficient for exploration


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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
        f"Using reward function: {'shaped (incremental)' if USE_SHAPED_REWARD else 'simple (sparse)'}"
    )
    print(f"Creating {N_ENVS} parallel environments...")
    env = make_vec_env(
        make_catan_env,
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,  # Use subprocesses for CPU-heavy environments
    )

    # Load or create model
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    if args.resume:
        checkpoint = get_latest_checkpoint()
        if checkpoint:
            print(f"Loading checkpoint: {checkpoint}")
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
        net_arch = [NEURONS_PER_LAYER] * NUM_LAYERS
        policy_kwargs = dict(net_arch=net_arch)

        # Create learning rate schedule
        lr_schedule = linear_schedule(INITIAL_LR, FINAL_LR)

        print(f"Creating new model with architecture: {net_arch}")
        print(
            f"PPO config: n_steps={N_STEPS}, batch_size={BATCH_SIZE}, ent_coef={ENT_COEF}"
        )
        print(f"Learning rate schedule: {INITIAL_LR:.2e} → {FINAL_LR:.2e}")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            learning_rate=lr_schedule,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=ENT_COEF,
            verbose=1,
            tensorboard_log=TENSORBOARD_DIR,
            seed=SEED,
            policy_kwargs=policy_kwargs,
            device=device,
        )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
    )

    # Train
    print(f"\nTraining for {args.timesteps:,} timesteps")
    print(f"With {N_ENVS} parallel environments (~{N_ENVS}x speedup)")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not args.resume,
        tb_log_name="MaskablePPO",
    )

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.zip")
    model.save(final_path)
    print(f"\nDone! Final model: {final_path}")

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


def make_catan_env():
    """Factory function to create a Catan environment for vectorization."""
    # Create fresh reward function instance for each environment
    reward_fn = ShapedRewardFunction() if USE_SHAPED_REWARD else simple_reward

    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
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
