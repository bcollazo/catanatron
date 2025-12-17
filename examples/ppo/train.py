"""
Restartable Stable Baselines3 training example with TensorBoard logging.

Features:
- Shaped reward function (incremental rewards for progress)
- GPU support (automatic detection)
- Checkpoint saving/loading for resumable training
- TensorBoard integration for monitoring

Configure by editing constants:
    NUM_LAYERS = 3              # Number of hidden layers
    NEURONS_PER_LAYER = 256     # Neurons in each layer
    N_STEPS = 2048              # PPO rollout steps
    BATCH_SIZE = 64             # PPO batch size
    USE_SHAPED_REWARD = True    # Incremental vs sparse rewards

Usage:
    # Start new training:
    python train.py

    # Resume from checkpoint:
    python train.py --resume

    # View TensorBoard:
    tensorboard --logdir ./tensorboard_logs
"""

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

import catanatron.gym
from catanatron import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import simple_reward
from shaped_reward import shaped_reward


# Configuration
SEED = 42
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TENSORBOARD_DIR = os.path.join(os.path.dirname(__file__), "tensorboard_logs")
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_FREQ = 10_000
NUM_LAYERS = 3
NEURONS_PER_LAYER = 256
USE_SHAPED_REWARD = True

# PPO parameters
N_STEPS = 2048  # Number of steps to collect before update (default: 2048)
BATCH_SIZE = 128  # Batch size for training (default: 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Seed everything for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Set device (GPU if available)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    # Create environment
    reward_fn = shaped_reward if USE_SHAPED_REWARD else simple_reward
    print(
        f"Using reward function: {'shaped (incremental)' if USE_SHAPED_REWARD else 'simple (sparse)'}"
    )
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [ValueFunctionPlayer(Color.RED)],
            "reward_function": reward_fn,
        },
    )
    env = ActionMasker(env, mask_fn)
    env.reset(seed=SEED)

    # Load or create model
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

        print(f"Creating new model with architecture: {net_arch}")
        print(f"PPO config: n_steps={N_STEPS}, batch_size={BATCH_SIZE}")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
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
    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=not args.resume,
        tb_log_name="MaskablePPO",
    )

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.zip")
    model.save(final_path)
    print(f"\nDone! Final model: {final_path}")


def mask_fn(env) -> np.ndarray:
    """Create action mask for valid actions."""
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])


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


if __name__ == "__main__":
    main()
