import argparse
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color

# Debug print
print("Attempting to import MyPlayer...")
try:
    from catanatron_experimental.my_player import MyPlayer
    print(f"MyPlayer imported successfully as: {MyPlayer}")
except ImportError as e:
    print(f"Failed to import MyPlayer: {e}")
    try:
        from catanatron_experimental.my_player import MyPlayer
        print(f"MyPlayer imported successfully from alternative location as: {MyPlayer}")
    except ImportError as e:
        print(f"Failed to import MyPlayer from alternative location: {e}")
        sys.exit(1)

from catanatron_gym.envs.catanatron_env import CatanatronEnv
from sb3_contrib.common.wrappers import ActionMasker

# Debug print
print(f"MyPlayer is: {MyPlayer}")

def train_model(episodes, timesteps_per_episode, model_path=None):
    player = MyPlayer(Color.RED, model_path)
    if player.model is None:
        dummy_game = Game([player, RandomPlayer(Color.BLUE)])
        player.initialize_model(dummy_game)

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        game = Game([player, RandomPlayer(Color.BLUE)])
        env = CatanatronEnv(config={
            "map_type": game.state.board.map.__class__.__name__,
            "discard_limit": game.state.discard_limit,
            "vps_to_win": game.vps_to_win,
            "num_players": len(game.state.colors),
            "player_colors": game.state.colors,
            "catan_map": game.state.board.map,
        })
        env = ActionMasker(env, player.mask_fn)
        player.model.set_env(env)
        player.model.learn(total_timesteps=timesteps_per_episode)

    save_path = model_path or "trained_model.zip"
    player.model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Catan AI model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--timesteps", type=int, default=1000, help="Timesteps per episode")
    parser.add_argument("--model", type=str, help="Path to load/save the model")
    args = parser.parse_args()

    train_model(args.episodes, args.timesteps, args.model)
