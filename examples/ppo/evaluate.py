"""
Evaluate a trained PPO agent against ValueFunctionPlayer using the simulator.

This script uses the play_batch method to run games in the simulator
(not the gym environment), which is much faster for evaluation.

Usage:
    # Evaluate final model:
    python evaluate.py --model-path checkpoints/final_model.zip

    # Evaluate with custom number of games:
    python evaluate.py --model-path checkpoints/final_model.zip --num-games 1000

    # Evaluate from a specific checkpoint:
    python evaluate.py --model-path checkpoints/rl_model_50000_steps.zip
"""

import argparse
import os

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from catanatron.features import create_sample, get_feature_ordering
from catanatron.models.player import Player, Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.cli.play import play_batch, GameConfigOptions
from catanatron.gym.envs.action_space import to_action_space, from_action_space

import catanatron.gym
from ppo_utils import autodetect_vecnormalize_path, make_catan_env


class PPOPlayer(Player):
    """Player that uses a trained PPO model to make decisions."""

    def __init__(
        self,
        color,
        model,
        env,
        map_type="MINI",
        debug=False,
    ):
        """
        Initialize PPO player.

        Args:
            color: Player color
            model: Trained MaskablePPO model
            env: Environment (DummyVecEnv or VecNormalize-wrapped)
            map_type: Map type for action space conversion
            player_colors: Tuple of player colors for action space conversion
            debug: Whether to print debug information
        """
        super().__init__(color, is_bot=True)
        self.model = model
        self.env = env
        self.map_type = map_type
        self.debug = debug
        self.decision_count = 0

    def decide(self, game, playable_actions):
        """
        Use PPO model to choose an action.

        Args:
            game: Current game state
            playable_actions: List of valid actions

        Returns:
            Selected action from playable_actions
        """
        # Get observation from the environment
        features = get_feature_ordering(len(game.state.colors), self.map_type)
        sample = create_sample(game, self.color)
        obs = np.array([sample[i] for i in features], dtype=np.float32)

        # Normalize observation if using VecNormalize
        # VecNormalize expects batched observations, so we need to add batch dimension
        if isinstance(self.env, VecNormalize):
            obs = np.expand_dims(obs, axis=0)  # Add batch dimension
            obs = self.env.normalize_obs(obs)
            obs = obs[0]  # Remove batch dimension

        # Create action mask
        valid_action_indices = sorted(
            [
                to_action_space(a, game.state.colors, self.map_type)
                for a in playable_actions
            ]
        )
        action_mask = np.zeros(self.env.action_space.n, dtype=bool)
        action_mask[valid_action_indices] = True

        # Predict action using the model
        # Note: predict expects a single observation (not batched for single prediction)
        action_idx, _ = self.model.predict(
            obs, action_masks=np.array([action_mask]), deterministic=True
        )

        # Convert action index back to catan action
        if isinstance(action_idx, np.ndarray):
            action_idx_int = (
                int(action_idx.item()) if action_idx.ndim == 0 else int(action_idx[0])
            )
        else:
            action_idx_int = int(action_idx)
        selected_action = from_action_space(
            action_idx_int, self.color, game.state.colors, self.map_type
        )

        # Find and return the matching action from playable_actions
        for action in playable_actions:
            if (
                action.action_type == selected_action.action_type
                and action.value == selected_action.value
            ):
                if self.debug and self.decision_count < 10:
                    print(
                        f"Decision {self.decision_count}: Selected {action.action_type} (index {action_idx_int})"
                    )
                self.decision_count += 1
                return action

        # Fallback: if model predicted invalid action, return first playable action
        print(
            f"Warning: Model predicted invalid action {selected_action.action_type}, using fallback"
        )
        if self.debug:
            print(
                f"  Valid actions were: {[a.action_type for a in playable_actions[:5]]}"
            )
        self.decision_count += 1
        return playable_actions[0]


def evaluate_model(
    model_path,
    vecnorm_path=None,
    num_games=100,
    map_type="MINI",
    vps_to_win=6,
):
    """
    Evaluate a trained PPO model against ValueFunctionPlayer.

    Args:
        model_path: Path to trained model (.zip)
        vecnorm_path: Path to VecNormalize stats (.pkl)
        num_games: Number of games to play
        map_type: Map type
        vps_to_win: Victory points to win

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating PPO model: {model_path}")
    print(f"Playing {num_games} games against ValueFunctionPlayer...")

    # Load model
    config = {
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "use_shaped_reward": True,
    }
    temp_env = DummyVecEnv([lambda: make_catan_env(config)])

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from: {vecnorm_path}")
        temp_env = VecNormalize.load(vecnorm_path, temp_env)
        temp_env.training = False
        temp_env.norm_reward = False

    print(f"Loading model from: {model_path}")
    model = MaskablePPO.load(model_path, env=temp_env)

    # Create players
    ppo_player = PPOPlayer(
        Color.BLUE,
        model,
        temp_env,
        map_type=map_type,
        debug=True,  # Enable debug output
    )
    value_player = ValueFunctionPlayer(Color.RED)
    players = [ppo_player, value_player]

    game_config = GameConfigOptions(
        map_type=map_type,
        vps_to_win=vps_to_win,
    )

    play_batch(
        num_games=num_games,
        players=players,
        game_config=game_config,
        quiet=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent against ValueFunctionPlayer"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file (.zip)",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=str,
        help="Path to VecNormalize stats (.pkl). If not provided, will auto-detect.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="MINI",
        help="Map type (default: MINI)",
    )
    parser.add_argument(
        "--vps-to-win",
        type=int,
        default=6,
        help="Victory points to win (default: 6)",
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

    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        vecnorm_path=vecnorm_path,
        num_games=args.num_games,
        map_type=args.map_type,
        vps_to_win=args.vps_to_win,
    )


if __name__ == "__main__":
    main()
