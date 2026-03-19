"""Collect supervised-learning data for the PlacementAgent.

Runs full games between two catanatron bots, records every placement
decision made by the Blue player (obs, mask, action from Blue's
perspective), and labels each with whether Blue won.

Both players use real strategies, so the placement data reflects
genuinely good (or at least intentional) decisions -- not random noise.

Output: a .npz file containing:
    obs     (N, 1258)  float32  -- board observation at decision time
    masks   (N, 245)   float32  -- valid-action mask
    actions (N,)       int64    -- capstone action index chosen
    won     (N,)       float32  -- 1.0 if Blue won, 0.0 otherwise

Usage:
    python capstone_agent/collect_placement_data.py --games 5000

    python capstone_agent/collect_placement_data.py --games 1000 \
        --blue alphabeta --enemy alphabeta \
        --out capstone_agent/placement_data_ab_vs_ab.npz
"""

import sys
import os
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from action_map import validate as validate_action_mapping
from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.gym.envs.capstone_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    to_action_space as catanatron_to_action_space,
)
from catanatron.gym.envs.action_translator import (
    batch_catanatron_to_capstone,
    catanatron_action_to_capstone_index,
)


BOT_TYPES = {
    "random": lambda color: RandomPlayer(color),
    "weighted": lambda color: WeightedRandomPlayer(color),
    "vp": lambda color: VictoryPointPlayer(color),
    "alphabeta": lambda color: AlphaBetaPlayer(color),
    "alphabeta-prune": lambda color: AlphaBetaPlayer(color, prunning=True),
    "same-turn-ab": lambda color: SameTurnAlphaBetaPlayer(color),
    "value": lambda color: ValueFunctionPlayer(color),
}


def _make_action_mask(game, color):
    """Build a 245-dim binary mask of valid capstone actions."""
    catanatron_indices = [
        catanatron_to_action_space(a) for a in game.playable_actions
    ]
    capstone_indices = batch_catanatron_to_capstone(catanatron_indices)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for idx in capstone_indices:
        mask[idx] = 1.0
    return mask


def collect(
    num_games: int,
    blue_type: str = "alphabeta",
    enemy_type: str = "alphabeta",
    verbose: bool = False,
):
    validate_action_mapping()

    for name, registry in [("blue", blue_type), ("enemy", enemy_type)]:
        if registry not in BOT_TYPES:
            available = ", ".join(sorted(BOT_TYPES))
            raise ValueError(
                f"Unknown {name} type {registry!r}. Choose from: {available}"
            )

    self_color = Color.BLUE
    opp_color = Color.RED

    all_obs, all_masks, all_actions = [], [], []
    game_indices = []
    game_outcomes = []

    for g in range(num_games):
        blue_player = BOT_TYPES[blue_type](self_color)
        red_player = BOT_TYPES[enemy_type](opp_color)
        game = Game(players=[blue_player, red_player])

        placement_obs, placement_masks, placement_actions = [], [], []

        while (
            game.winning_color() is None
            and game.state.num_turns < TURNS_LIMIT
        ):
            current_color = game.state.current_color()
            is_placement = game.state.is_initial_build_phase

            if is_placement and current_color == self_color:
                obs = np.array(
                    get_capstone_observation(game, self_color, opp_color),
                    dtype=np.float32,
                )
                mask = _make_action_mask(game, self_color)
                placement_obs.append(obs)
                placement_masks.append(mask)

            action_record = game.play_tick()
            action = action_record.action

            if is_placement and current_color == self_color:
                cap_idx = catanatron_action_to_capstone_index(action)
                placement_actions.append(cap_idx)

        won = game.winning_color() == self_color

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
        description="Collect placement training data from bot-vs-bot games"
    )
    parser.add_argument(
        "--games", type=int, default=5000, help="Number of games to play"
    )
    parser.add_argument(
        "--blue",
        type=str,
        default="alphabeta",
        choices=sorted(BOT_TYPES),
        help="Bot type for Blue (the player we learn from)",
    )
    parser.add_argument(
        "--enemy",
        type=str,
        default="alphabeta",
        choices=sorted(BOT_TYPES),
        help="Bot type for Red (the opponent)",
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
        f"(blue={args.blue} vs enemy={args.enemy}) ..."
    )
    obs, masks, actions, won = collect(
        args.games,
        blue_type=args.blue,
        enemy_type=args.enemy,
        verbose=args.verbose,
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
