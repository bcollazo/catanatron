"""
Record videos of a trained PPO agent playing Catanatron.

Usage:
    # Record from a local model:
    python record_video.py --model-path checkpoints/final_model.zip

    # Auto-detect VecNormalize stats (looks for matching .pkl file):
    python record_video.py --model-path checkpoints/rl_model_50000_steps.zip

    # Custom number of episodes:
    python record_video.py --model-path checkpoints/final_model.zip --num-episodes 10

    # Custom output directory:
    python record_video.py --model-path checkpoints/final_model.zip --output-dir my_videos/
"""

import argparse
import os
from pathlib import Path

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

import catanatron.gym
from ppo_utils import autodetect_vecnormalize_path, make_catan_env


def record_videos(
    model_path,
    vecnorm_path=None,
    config=None,
    output_dir="videos",
    num_episodes=3,
):
    """Record videos of the agent playing and save to local filesystem."""
    # Default config if not provided
    if config is None:
        config = {
            "map_type": "MINI",
            "vps_to_win": 6,
            "use_shaped_reward": True,
        }

    print(f"\nRecording {num_episodes} episodes...")
    print(f"Model: {model_path}")
    if vecnorm_path:
        print(f"VecNormalize stats: {vecnorm_path}")

    # Create environment
    env = DummyVecEnv([lambda: make_catan_env(config)])

    # Load VecNormalize stats if available
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # Don't update normalization stats
        env.norm_reward = False  # Don't normalize rewards during evaluation

    # Wrap with video recorder
    os.makedirs(output_dir, exist_ok=True)
    env = VecVideoRecorder(
        env,
        output_dir,
        record_video_trigger=lambda x: x % 1 == 0,  # Record every episode
        video_length=500,  # Max steps per video
        name_prefix="catanatron",
    )

    # Load model
    print("\nLoading model...")
    model = MaskablePPO.load(model_path, env=env)

    # Run episodes
    obs = env.reset()
    episode_count = 0
    total_reward = 0
    episode_rewards = []

    while episode_count < num_episodes:
        action_masks = np.array([env.envs[0].unwrapped.action_masks()])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        if done[0]:
            episode_count += 1
            episode_rewards.append(total_reward)
            print(f"  Episode {episode_count}: Reward = {total_reward:.2f}")
            total_reward = 0

    env.close()

    # Find generated videos
    video_dir = Path(output_dir)
    video_files = sorted(video_dir.glob("*.mp4"))

    print(f"\nRecorded {len(video_files)} videos to {output_dir}/")
    print(
        f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )

    return video_files


def main():
    parser = argparse.ArgumentParser(
        description="Record videos of a trained PPO agent playing Catanatron"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local model file (.zip)",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=str,
        help="Path to VecNormalize stats (.pkl). If not provided, will auto-detect.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="videos", help="Directory to save videos"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=3, help="Number of episodes to record"
    )

    args = parser.parse_args()

    # Validate model path exists
    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")

    # Auto-detect VecNormalize stats if not provided
    vecnorm_path, auto_detected = autodetect_vecnormalize_path(
        args.model_path, args.vecnorm_path
    )
    if auto_detected:
        print(f"Auto-detected VecNormalize stats: {vecnorm_path}")
    elif not vecnorm_path:
        print("No VecNormalize stats found (will train without normalization)")

    # Record videos
    record_videos(
        model_path=args.model_path,
        vecnorm_path=vecnorm_path,
        config=None,  # Use defaults
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()
