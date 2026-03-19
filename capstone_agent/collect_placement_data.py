"""Collect supervised-learning data for the PlacementAgent.

Plays many games using a random policy for Blue, records every placement
decision (obs, mask, action) during the initial-build phase, and labels
each with whether Blue won.  The opponent can be any catanatron bot.

Output: a .npz file containing:
    obs     (N, 1258)  float32  -- board observation at decision time
    masks   (N, 245)   float32  -- valid-action mask
    actions (N,)       int64    -- action index taken
    won     (N,)       float32  -- 1.0 if Blue won, 0.0 otherwise

Usage:
    python capstone_agent/collect_placement_data.py --games 5000 \
        --out capstone_agent/placement_data.npz

    python capstone_agent/collect_placement_data.py --games 1000 \
        --enemy alphabeta --out capstone_agent/placement_data_ab.npz
"""

import sys
import os
import argparse

import numpy as np
import gymnasium

sys.path.insert(0, os.path.dirname(__file__))

from action_map import validate as validate_action_mapping
import catanatron.gym  # noqa: F401  (registers the env)
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer

ENEMY_TYPES = {
    "random": lambda: RandomPlayer(Color.RED),
    "weighted": lambda: WeightedRandomPlayer(Color.RED),
    "vp": lambda: VictoryPointPlayer(Color.RED),
    "alphabeta": lambda: AlphaBetaPlayer(Color.RED),
    "alphabeta-prune": lambda: AlphaBetaPlayer(Color.RED, prunning=True),
    "same-turn-ab": lambda: SameTurnAlphaBetaPlayer(Color.RED),
    "value": lambda: ValueFunctionPlayer(Color.RED),
}


def collect(num_games: int, enemy: str = "random", verbose: bool = False):
    """Run *num_games* with random actions and return placement data."""
    validate_action_mapping()

    if enemy not in ENEMY_TYPES:
        available = ", ".join(sorted(ENEMY_TYPES))
        raise ValueError(f"Unknown enemy {enemy!r}. Choose from: {available}")

    enemy_factory = ENEMY_TYPES[enemy]
    env = gymnasium.make(
        "catanatron/CapstoneCatanatron-v0",
        config={"enemies": [enemy_factory()]},
    )

    all_obs, all_masks, all_actions = [], [], []
    game_indices = []
    game_outcomes = []

    for g in range(num_games):
        env.unwrapped.enemies = [enemy_factory()]
        obs, info = env.reset()
        mask = info["action_mask"]
        is_placement = info["is_initial_build_phase"]

        placement_obs, placement_masks, placement_actions = [], [], []

        done = False
        while not done:
            valid = np.where(mask > 0.5)[0]
            action = int(np.random.choice(valid))

            if is_placement:
                placement_obs.append(obs.astype(np.float32))
                placement_masks.append(mask.astype(np.float32))
                placement_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            is_placement = info["is_initial_build_phase"]
            done = terminated or truncated

        won = reward > 0

        for i in range(len(placement_obs)):
            all_obs.append(placement_obs[i])
            all_masks.append(placement_masks[i])
            all_actions.append(placement_actions[i])
            game_indices.append(g)
        game_outcomes.append(won)

        if verbose and (g + 1) % 500 == 0:
            win_rate = sum(game_outcomes) / len(game_outcomes)
            print(
                f"  {g + 1}/{num_games} games  "
                f"({len(all_obs)} samples, win rate {win_rate:.2%})"
            )

    won_arr = np.array(
        [float(game_outcomes[gi]) for gi in game_indices], dtype=np.float32
    )

    return (
        np.array(all_obs, dtype=np.float32),
        np.array(all_masks, dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
        won_arr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect placement training data from games"
    )
    parser.add_argument(
        "--games", type=int, default=5000, help="Number of games to play"
    )
    parser.add_argument(
        "--enemy",
        type=str,
        default="random",
        choices=sorted(ENEMY_TYPES),
        help="Opponent bot type",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="capstone_agent/placement_data.npz",
        help="Output .npz path",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(
        f"Collecting placement data from {args.games} games "
        f"(enemy={args.enemy}) ..."
    )
    obs, masks, actions, won = collect(
        args.games, enemy=args.enemy, verbose=args.verbose
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, obs=obs, masks=masks, actions=actions, won=won)

    n_wins = int(won.sum())
    print(
        f"Saved {len(obs)} samples to {args.out}  "
        f"({n_wins} from wins, {len(obs) - n_wins} from losses)"
    )


if __name__ == "__main__":
    main()
