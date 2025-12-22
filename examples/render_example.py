"""
Example script that plays a game and renders it with video recording.

Uses gymnasium's RecordVideo wrapper to automatically record gameplay videos.

Usage:
    # Record a game to video
    python render_example.py --render-style video

    # Record to custom folder
    python render_example.py --render-style video --video-folder ./my_videos

    # Save a replay to DB (prints link on completion)
    python render_example.py --render-style db
"""

import random
import argparse
import gymnasium
from gymnasium.wrappers import RecordVideo

import catanatron.gym  # Register the environment
from catanatron import Color
from catanatron.players.value import ValueFunctionPlayer


def main():
    parser = argparse.ArgumentParser(description="Play and record Catanatron games")
    parser.add_argument(
        "--render-style",
        type=str,
        default="video",
        choices=["video", "db"],
        help="Render style to use: video (rgb_array) or db (replay link).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="./videos",
        help="Folder to save videos (default: ./videos)",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI"],
        help="Map type to use (default: MINI)",
    )
    parser.add_argument(
        "--vps-to-win",
        type=int,
        default=10,
        help="Victory points needed to win (default: 6)",
    )
    args = parser.parse_args()

    render_mode = "rgb_array" if args.render_style == "video" else "db"

    # Create env with rendering enabled
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "render_mode": render_mode,
            "render_scale": 2.0,
            "map_type": args.map_type,
            "vps_to_win": args.vps_to_win,
            "enemies": [
                ValueFunctionPlayer(Color.RED),
                ValueFunctionPlayer(Color.ORANGE),
                ValueFunctionPlayer(Color.WHITE),
            ],
        },
    )

    # Wrap with RecordVideo when rendering video
    if args.render_style == "video":
        env = RecordVideo(
            env,
            video_folder=args.video_folder,
            name_prefix="catan-game",
            episode_trigger=lambda x: True,  # Record every episode
        )
        print(f"Recording video to: {args.video_folder}")
    else:
        print("Saving replay to DB (render() will write steps).")

    observation, info = env.reset()
    done = False
    step = 0

    print(f"Starting game with {args.map_type} map, {args.vps_to_win} VPs to win...")

    while not done:
        # Get valid actions
        valid_actions = info["valid_actions"]

        # Take first valid action (random)
        action = random.choice(valid_actions)

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if args.render_style == "db":
            env.render()

        step += 1
        if step % 10 == 0:
            print(f"Step {step}...")

    print(f"Game finished after {step} steps")

    if args.render_style == "video":
        print(f"Video saved to {args.video_folder}")

    env.close()


if __name__ == "__main__":
    main()
