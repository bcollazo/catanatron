"""Benchmark placement strategies against AlphaBeta.

Runs two groups of games:
  1. **Baseline** – Blue = full AlphaBeta (placement + rest of game)
  2. **Test**     – Blue = our placement strategy for initial build,
                    then AlphaBeta for the rest of the game

Red is always AlphaBeta.  The only independent variable is the
initial-placement strategy.

Usage:
    python capstone_agent/benchmark_placement.py --games 100

    python capstone_agent/benchmark_placement.py --games 200 \
        --strategy model --placement-model capstone_agent/models/placement_model.pt

    python capstone_agent/benchmark_placement.py --games 200 \
        --strategy random
"""

import sys
import os
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.gym.envs.capstone_env import (
    ACTION_SPACE_SIZE,
    to_action_space as capstone_action_index,
)
from catanatron.gym.envs.action_translator import capstone_to_action
from PlacementAgent import make_placement_agent

from CONSTANTS import PLACEMENT_AGENT_HIDDEN_SIZE


def _make_action_mask(game):
    """Build a 245-dim binary mask of valid capstone actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for a in game.playable_actions:
        mask[capstone_action_index(a)] = 1.0
    return mask


class HybridPlayer(Player):
    """Uses a PlacementAgent for initial build, AlphaBeta for everything else."""

    def __init__(self, color, placement_agent, ab_depth=2):
        super().__init__(color, is_bot=True)
        self.placement_agent = placement_agent
        self.ab = AlphaBetaPlayer(color, depth=ab_depth)

    def decide(self, game, playable_actions):
        if game.state.is_initial_build_phase:
            return self._placement_decide(game, playable_actions)
        return self.ab.decide(game, playable_actions)

    def _placement_decide(self, game, playable_actions):
        opp_color = [c for c in game.state.colors if c != self.color][0]
        obs = np.array(
            get_capstone_observation(game, self.color, opp_color),
            dtype=np.float32,
        )
        mask = _make_action_mask(game)
        action_idx, _, _ = self.placement_agent.select_action(obs, mask)
        return capstone_to_action(action_idx, playable_actions)


def run_group(
    label: str,
    blue_factory,
    num_games: int,
    verbose: bool = False,
):
    """Play *num_games* and return (wins, losses, draws, elapsed_seconds)."""
    wins, losses, draws = 0, 0, 0
    t0 = time.time()

    for g in range(1, num_games + 1):
        blue = blue_factory()
        red = AlphaBetaPlayer(Color.RED)
        game = Game(players=[blue, red])

        winner = game.play()

        if winner == Color.BLUE:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        if verbose and g % 50 == 0:
            elapsed = time.time() - t0
            wr = wins / g
            print(
                f"  [{label}] {g}/{num_games}  "
                f"W={wins} L={losses} D={draws}  "
                f"win_rate={wr:.1%}  ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    return wins, losses, draws, elapsed


def print_summary(label, wins, losses, draws, n, elapsed):
    wr = wins / n if n else 0
    lo = wr - 1.96 * (wr * (1 - wr) / n) ** 0.5
    hi = wr + 1.96 * (wr * (1 - wr) / n) ** 0.5
    print(
        f"  {label:24s}  "
        f"{wins:4d}W / {losses:4d}L / {draws:4d}D  "
        f"win_rate={wr:.1%}  "
        f"95% CI=[{lo:.1%}, {hi:.1%}]  "
        f"({elapsed:.0f}s)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark placement strategies vs AlphaBeta"
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Games per group (baseline + test)",
    )
    parser.add_argument(
        "--strategy", type=str, default="random",
        choices=["random", "model"],
        help="Placement strategy to test against the AlphaBeta baseline",
    )
    parser.add_argument(
        "--placement-model", type=str, default=None,
        help="Path to trained placement model weights (required for --strategy model)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the AlphaBeta-vs-AlphaBeta baseline group",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.strategy == "model" and args.placement_model is None:
        parser.error("--placement-model is required when --strategy is 'model'")

    n = args.games

    # ── baseline: pure AlphaBeta ──────────────────────────────────
    if not args.skip_baseline:
        print(f"=== Baseline: AlphaBeta placement ({n} games) ===")
        bw, bl, bd, bt = run_group(
            "baseline",
            lambda: AlphaBetaPlayer(Color.BLUE),
            n,
            verbose=args.verbose,
        )
    else:
        bw, bl, bd, bt = 0, 0, 0, 0.0

    # ── test: our placement strategy + AlphaBeta rest ─────────────
    placement_agent = make_placement_agent(
        args.strategy,
        hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
    )
    if args.placement_model:
        placement_agent.load(args.placement_model)

    strategy_label = args.strategy
    if args.placement_model:
        strategy_label += f" ({os.path.basename(args.placement_model)})"

    print(f"\n=== Test: {strategy_label} placement + AlphaBeta rest ({n} games) ===")
    tw, tl, td, tt = run_group(
        strategy_label,
        lambda: HybridPlayer(Color.BLUE, placement_agent),
        n,
        verbose=args.verbose,
    )

    # ── summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    if not args.skip_baseline:
        print_summary("AlphaBeta (baseline)", bw, bl, bd, n, bt)
    print_summary(strategy_label, tw, tl, td, n, tt)

    if not args.skip_baseline and n > 0:
        delta = (tw / n) - (bw / n)
        print(f"\n  Delta (test - baseline): {delta:+.1%}")


if __name__ == "__main__":
    main()
