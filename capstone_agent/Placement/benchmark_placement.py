"""Benchmark placement strategies against AlphaBeta.

Runs two groups of games:
  1. **Baseline** – target color = full AlphaBeta (placement + rest of game)
  2. **Test**     – target color = our placement strategy for initial build,
                    then AlphaBeta for the rest of the game

The opponent is always AlphaBeta.  The only independent variable is the
target player's initial-placement strategy.

Usage:
    python capstone_agent/benchmark_placement.py --games 100

    python capstone_agent/benchmark_placement.py --games 200 \
        --strategy model --placement-model capstone_agent/models/placement_model.pt

    python capstone_agent/benchmark_placement.py --games 200 \
        --strategy random
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from catanatron.game import Game
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
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


def _append_jsonl(path: str, row: dict):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _confidence_interval_bounds(wins: int, games: int):
    if games <= 0:
        return 0.0, 0.0
    win_rate = wins / games
    margin = 1.96 * (win_rate * (1 - win_rate) / games) ** 0.5
    return win_rate - margin, win_rate + margin


def _fixed_map(board_seed: int = 42):
    random.seed(board_seed)
    return CatanMap.from_template(BASE_MAP_TEMPLATE)


def _binomial_test_greater(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0

    try:
        from scipy.stats import binomtest

        return float(binomtest(successes, trials, p=0.5, alternative="greater").pvalue)
    except Exception:
        tail_count = sum(math.comb(trials, k) for k in range(successes, trials + 1))
        return tail_count / (1 << trials)


def _paired_mcnemar_counts(baseline_wins: list[bool], test_wins: list[bool]):
    if len(baseline_wins) != len(test_wins):
        raise ValueError("McNemar inputs must have the same length")

    baseline_only = 0
    test_only = 0
    both_won = 0
    neither_won = 0

    for baseline_win, test_win in zip(baseline_wins, test_wins):
        if baseline_win and test_win:
            both_won += 1
        elif baseline_win and not test_win:
            baseline_only += 1
        elif test_win and not baseline_win:
            test_only += 1
        else:
            neither_won += 1

    discordant = baseline_only + test_only
    return {
        "baseline_only_wins": baseline_only,
        "test_only_wins": test_only,
        "both_won": both_won,
        "neither_won": neither_won,
        "discordant": discordant,
        "mcnemar_pvalue": _binomial_test_greater(test_only, discordant),
    }


def _log_benchmark_group(
    results_log: str | None,
    *,
    label: str,
    strategy: str,
    placement_model: str | None,
    temperature: float,
    board_seed: int,
    game_seed_start: int,
    games: int,
    wins: int,
    losses: int,
    draws: int,
    elapsed_seconds: float,
    target_color: Color = Color.BLUE,
):
    if results_log is None:
        return
    ci_lo, ci_hi = _confidence_interval_bounds(wins, games)
    _append_jsonl(
        results_log,
        {
            "event": "benchmark_group",
            "label": label,
            "strategy": strategy,
            "placement_model": placement_model,
            "temperature": temperature,
            "board_seed": board_seed,
            "game_seed_start": game_seed_start,
            "games": games,
            "target_color": target_color.value,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / games if games else 0.0,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "elapsed_seconds": elapsed_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def _log_paired_results(
    results_log: str | None,
    *,
    strategy: str,
    placement_model: str | None,
    temperature: float,
    board_seed: int,
    game_seed_start: int,
    games: int,
    paired_counts: dict,
    target_color: Color = Color.BLUE,
):
    if results_log is None:
        return

    row = {
        "event": "benchmark_paired_mcnemar",
        "strategy": strategy,
        "placement_model": placement_model,
        "temperature": temperature,
        "board_seed": board_seed,
        "game_seed_start": game_seed_start,
        "games": games,
        "target_color": target_color.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row.update(paired_counts)
    _append_jsonl(results_log, row)


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
        action_idx, _, _ = self.placement_agent.select_action(
            obs,
            mask,
            game=game,
            playable_actions=playable_actions,
        )
        return capstone_to_action(action_idx, playable_actions)


def run_group(
    label: str,
    blue_factory,
    num_games: int,
    *,
    red_factory=None,
    target_color: Color = Color.BLUE,
    board_seed: int = 42,
    game_seed_start: int = 0,
    verbose: bool = False,
):
    """Play games and return wins, losses, draws, elapsed seconds, and win booleans."""
    wins, losses, draws = 0, 0, 0
    target_wins = []
    fixed_map = _fixed_map(board_seed)
    red_factory = red_factory or (lambda: AlphaBetaPlayer(Color.RED))
    t0 = time.time()

    for game_idx in range(num_games):
        g = game_idx + 1
        game_seed = game_seed_start + game_idx
        np.random.seed(game_seed % (2**32 - 1))

        blue = blue_factory()
        red = red_factory()
        game = Game(players=[blue, red], catan_map=fixed_map, seed=game_seed)

        winner = game.play()
        target_won = winner == target_color
        target_wins.append(target_won)

        if target_won:
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
    return wins, losses, draws, elapsed, target_wins


def print_summary(label, wins, losses, draws, n, elapsed):
    wr = wins / n if n else 0
    lo, hi = _confidence_interval_bounds(wins, n)
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
        choices=[
            "random",
            "model",
            "heuristic",
            "value_heuristic",
            "rollout_value",
            "rollout_value_selfish",
            "rollout_value_first_roll",
            "rollout_value_blend",
            "rollout_value_stable",
            "rollout_value_stable_first_roll",
            "rollout_value_stable_ab_opp",
            "beam_value",
        ],
        help="Placement strategy to test against the AlphaBeta baseline",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature for heuristic sampling. 0 = deterministic argmax.",
    )
    parser.add_argument(
        "--board-seed", type=int, default=42,
        help="Seed for the fixed board layout",
    )
    parser.add_argument(
        "--game-seed-start", type=int, default=0,
        help="First game seed. Game i uses game_seed_start + i.",
    )
    parser.add_argument(
        "--test-color",
        type=str,
        default="BLUE",
        choices=["BLUE", "RED"],
        help="Seat/color for the placement strategy under test.",
    )
    parser.add_argument(
        "--placement-model", type=str, default=None,
        help="Path to trained placement model weights (required for --strategy model)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the AlphaBeta-vs-AlphaBeta baseline group",
    )
    parser.add_argument(
        "--results-log", type=str, default=None,
        help="Optional JSONL file to append benchmark results",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.strategy == "model" and args.placement_model is None:
        parser.error("--placement-model is required when --strategy is 'model'")

    n = args.games
    test_color = Color[args.test_color]

    def alpha_beta_player(color):
        return AlphaBetaPlayer(color)

    # ── baseline: pure AlphaBeta ──────────────────────────────────
    if not args.skip_baseline:
        print(
            f"=== Baseline: {test_color.value} AlphaBeta placement "
            f"({n} games) ==="
        )
        bw, bl, bd, bt, baseline_wins = run_group(
            "baseline",
            lambda: alpha_beta_player(Color.BLUE),
            n,
            red_factory=lambda: alpha_beta_player(Color.RED),
            target_color=test_color,
            board_seed=args.board_seed,
            game_seed_start=args.game_seed_start,
            verbose=args.verbose,
        )
        _log_benchmark_group(
            args.results_log,
            label="baseline",
            strategy=args.strategy,
            placement_model=args.placement_model,
            temperature=args.temperature,
            board_seed=args.board_seed,
            game_seed_start=args.game_seed_start,
            games=n,
            wins=bw,
            losses=bl,
            draws=bd,
            elapsed_seconds=bt,
            target_color=test_color,
        )
    else:
        bw, bl, bd, bt = 0, 0, 0, 0.0
        baseline_wins = []

    # ── test: our placement strategy + AlphaBeta rest ─────────────
    if args.strategy == "heuristic":
        placement_agent = make_placement_agent(
            args.strategy,
            temperature=args.temperature,
        )
    else:
        placement_agent = make_placement_agent(
            args.strategy,
            hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
        )
    if args.placement_model:
        placement_agent.load(args.placement_model)

    strategy_label = args.strategy
    if args.strategy == "heuristic":
        strategy_label += f" (T={args.temperature:g})"
    if args.placement_model:
        strategy_label += f" ({os.path.basename(args.placement_model)})"

    print(
        f"\n=== Test: {test_color.value} {strategy_label} placement + "
        f"AlphaBeta rest ({n} games) ==="
    )
    if test_color == Color.BLUE:
        blue_factory = lambda: HybridPlayer(Color.BLUE, placement_agent)
        red_factory = lambda: alpha_beta_player(Color.RED)
    else:
        blue_factory = lambda: alpha_beta_player(Color.BLUE)
        red_factory = lambda: HybridPlayer(Color.RED, placement_agent)

    tw, tl, td, tt, test_wins = run_group(
        strategy_label,
        blue_factory,
        n,
        red_factory=red_factory,
        target_color=test_color,
        board_seed=args.board_seed,
        game_seed_start=args.game_seed_start,
        verbose=args.verbose,
    )
    _log_benchmark_group(
        args.results_log,
        label=strategy_label,
        strategy=args.strategy,
        placement_model=args.placement_model,
        temperature=args.temperature,
        board_seed=args.board_seed,
        game_seed_start=args.game_seed_start,
        games=n,
        wins=tw,
        losses=tl,
        draws=td,
        elapsed_seconds=tt,
        target_color=test_color,
    )

    # ── summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    if not args.skip_baseline:
        print_summary(f"{test_color.value} AlphaBeta (baseline)", bw, bl, bd, n, bt)
    print_summary(f"{test_color.value} {strategy_label}", tw, tl, td, n, tt)

    if not args.skip_baseline and n > 0:
        delta = (tw / n) - (bw / n)
        print(f"\n  Delta (test - baseline): {delta:+.1%}")

        paired_counts = _paired_mcnemar_counts(baseline_wins, test_wins)
        _log_paired_results(
            args.results_log,
            strategy=args.strategy,
            placement_model=args.placement_model,
            temperature=args.temperature,
            board_seed=args.board_seed,
            game_seed_start=args.game_seed_start,
            games=n,
            paired_counts=paired_counts,
            target_color=test_color,
        )
        print(
            "  McNemar exact test: "
            f"b={paired_counts['baseline_only_wins']}, "
            f"c={paired_counts['test_only_wins']}, "
            f"p={paired_counts['mcnemar_pvalue']:.6g}"
        )
        print(
            "  Test strategy wins "
            f"{paired_counts['test_only_wins']}/{paired_counts['discordant']} "
            "discordant pairs"
        )


if __name__ == "__main__":
    main()
