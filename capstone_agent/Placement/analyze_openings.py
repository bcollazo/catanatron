"""Compare AlphaBeta and heuristic opening placements on a fixed board.

Run from the repository root:

    python capstone_agent/Placement/analyze_openings.py

By default this stops after the initial placement phase. Pass ``--outcomes`` to
continue each paired game to completion and print winner summaries.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CAPSTONE_DIR = SCRIPT_DIR.parent
REPO_ROOT = CAPSTONE_DIR.parent
for path in (REPO_ROOT, CAPSTONE_DIR, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from capstone_agent.Placement.PlacementAgent import make_placement_agent
from catanatron.game import Game, TURNS_LIMIT
from catanatron.gym.envs.action_translator import (
    capstone_to_action,
    catanatron_action_to_capstone_index,
)
from catanatron.gym.envs.capstone_env import ACTION_SPACE_SIZE
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.models.enums import RESOURCES, SETTLEMENT, Action, ActionType
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, LandTile, number_probability
from catanatron.models.player import Color, Player
from catanatron.players.minimax import AlphaBetaPlayer


PIP_COUNTS = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
}


@dataclass
class OpeningDecision:
    ply: int
    color: Color
    policy: str
    action_type: str
    capstone_action: int | None
    action_value: object
    detail: str


@dataclass
class GameReport:
    label: str
    game_seed: int
    winner: Color | None
    truncated: bool
    decisions: list[OpeningDecision]


def fixed_map(board_seed: int) -> CatanMap:
    """Build one deterministic board that is reused across the seed range."""

    random.seed(board_seed)
    return CatanMap.from_template(BASE_MAP_TEMPLATE)


def make_action_mask(playable_actions: Iterable[Action]) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action in playable_actions:
        mask[catanatron_action_to_capstone_index(action)] = 1.0
    return mask


def other_color(game: Game, color: Color) -> Color:
    return next(candidate for candidate in game.state.colors if candidate != color)


def port_label(catan_map: CatanMap, node_id: int) -> str:
    for resource, node_ids in catan_map.port_nodes.items():
        if node_id in node_ids:
            if resource is None:
                return "3:1"
            return f"2:1 {resource}"
    return "none"


def format_node(catan_map: CatanMap, node_id: int) -> str:
    tiles = catan_map.adjacent_tiles.get(node_id, [])
    tile_parts = []
    total_pips = 0

    for tile in sorted(tiles, key=lambda item: item.id):
        if not isinstance(tile, LandTile) or tile.resource is None:
            tile_parts.append("DESERT")
            continue
        pips = PIP_COUNTS.get(tile.number, 0)
        total_pips += pips
        tile_parts.append(f"{tile.resource}:{tile.number}({pips}p)")

    production = catan_map.node_production.get(node_id, {})
    prod_parts = [
        f"{resource}={production[resource]:.3f}"
        for resource in RESOURCES
        if resource in production
    ]
    total_prob = sum(production.values())

    return (
        f"node={node_id} "
        f"tiles=[{', '.join(tile_parts) or 'none'}] "
        f"prod=[{', '.join(prod_parts) or 'none'}] "
        f"total={total_prob:.3f}/{total_pips}p "
        f"port={port_label(catan_map, node_id)}"
    )


def format_road(catan_map: CatanMap, game: Game, action: Action) -> str:
    edge = tuple(sorted(action.value))
    settlements = game.state.buildings_by_color[action.color][SETTLEMENT]
    anchor = settlements[-1] if settlements else None

    far_node = None
    if anchor is not None:
        if edge[0] == anchor:
            far_node = edge[1]
        elif edge[1] == anchor:
            far_node = edge[0]

    if far_node is None:
        endpoints = " | ".join(format_node(catan_map, node_id) for node_id in edge)
        return f"edge={edge} endpoints=({endpoints})"

    return (
        f"edge={edge} "
        f"from={anchor} "
        f"toward=({format_node(catan_map, far_node)})"
    )


def describe_opening_action(game: Game, action: Action) -> str:
    catan_map = game.state.board.map
    if action.action_type == ActionType.BUILD_SETTLEMENT:
        return format_node(catan_map, int(action.value))
    if action.action_type == ActionType.BUILD_ROAD:
        return format_road(catan_map, game, action)
    return repr(action.value)


def trace_decision(game: Game, action: Action, policy: str) -> OpeningDecision:
    try:
        capstone_action = catanatron_action_to_capstone_index(action)
    except Exception:
        capstone_action = None

    return OpeningDecision(
        ply=len(game.state.action_records) + 1,
        color=action.color,
        policy=policy,
        action_type=action.action_type.value,
        capstone_action=capstone_action,
        action_value=action.value,
        detail=describe_opening_action(game, action),
    )


class TracedAlphaBetaPlayer(Player):
    def __init__(self, color: Color, *, depth: int, policy_label: str):
        super().__init__(color, is_bot=True)
        self.policy_label = policy_label
        self.alpha_beta = AlphaBetaPlayer(color, depth=depth)
        self.opening_decisions: list[OpeningDecision] = []

    def decide(self, game: Game, playable_actions: list[Action]) -> Action:
        action = self.alpha_beta.decide(game, playable_actions)
        if game.state.is_initial_build_phase:
            self.opening_decisions.append(
                trace_decision(game, action, self.policy_label)
            )
        return action

    def reset_state(self):
        self.opening_decisions.clear()
        if hasattr(self.alpha_beta, "reset_state"):
            self.alpha_beta.reset_state()


class TracedPlacementPlayer(Player):
    def __init__(
        self,
        color: Color,
        *,
        strategy: str,
        alpha_beta_depth: int,
        temperature: float,
    ):
        super().__init__(color, is_bot=True)
        self.strategy = strategy
        if strategy == "heuristic":
            self.policy_label = f"HeuristicPlacementAgent(T={temperature:g})"
            self.placement_agent = make_placement_agent(
                strategy,
                temperature=temperature,
            )
        else:
            self.policy_label = strategy
            self.placement_agent = make_placement_agent(strategy)
        self.alpha_beta = AlphaBetaPlayer(color, depth=alpha_beta_depth)
        self.opening_decisions: list[OpeningDecision] = []

    def decide(self, game: Game, playable_actions: list[Action]) -> Action:
        if game.state.is_initial_build_phase:
            action = self._placement_decide(game, playable_actions)
            self.opening_decisions.append(
                trace_decision(game, action, self.policy_label)
            )
            return action
        return self.alpha_beta.decide(game, playable_actions)

    def _placement_decide(self, game: Game, playable_actions: list[Action]) -> Action:
        observation = np.asarray(
            get_capstone_observation(game, self.color, other_color(game, self.color)),
            dtype=np.float32,
        )
        mask = make_action_mask(playable_actions)
        capstone_action, _, _ = self.placement_agent.select_action(
            observation,
            mask,
            game=game,
            playable_actions=playable_actions,
        )
        return capstone_to_action(capstone_action, playable_actions)

    def reset_state(self):
        self.opening_decisions.clear()
        if hasattr(self.placement_agent, "reset_state"):
            self.placement_agent.reset_state()
        if hasattr(self.alpha_beta, "reset_state"):
            self.alpha_beta.reset_state()


def sorted_decisions(players: Iterable[Player]) -> list[OpeningDecision]:
    decisions: list[OpeningDecision] = []
    for player in players:
        decisions.extend(getattr(player, "opening_decisions", []))
    return sorted(decisions, key=lambda decision: decision.ply)


def play_report(
    *,
    label: str,
    game_seed: int,
    catan_map: CatanMap,
    players: list[Player],
    outcomes: bool,
    max_turns: int,
) -> GameReport:
    random.seed(game_seed)
    np.random.seed(game_seed % (2**32 - 1))

    game = Game(players=players, catan_map=catan_map, seed=game_seed)
    while game.state.is_initial_build_phase and game.winning_color() is None:
        game.play_tick()

    truncated = False
    if outcomes:
        while game.winning_color() is None and game.state.num_turns < max_turns:
            game.play_tick()
        truncated = game.winning_color() is None and game.state.num_turns >= max_turns

    return GameReport(
        label=label,
        game_seed=game_seed,
        winner=game.winning_color(),
        truncated=truncated,
        decisions=sorted_decisions(players),
    )


def run_pair(args, game_seed: int, catan_map: CatanMap) -> tuple[GameReport, GameReport]:
    baseline_players: list[Player] = [
        TracedAlphaBetaPlayer(
            Color.BLUE,
            depth=args.alpha_beta_depth,
            policy_label=f"AlphaBeta(depth={args.alpha_beta_depth})",
        ),
        TracedAlphaBetaPlayer(
            Color.RED,
            depth=args.alpha_beta_depth,
            policy_label=f"AlphaBeta(depth={args.alpha_beta_depth})",
        ),
    ]
    heuristic_players: list[Player] = [
        TracedPlacementPlayer(
            Color.BLUE,
            strategy=args.strategy,
            alpha_beta_depth=args.alpha_beta_depth,
            temperature=args.temperature,
        ),
        TracedAlphaBetaPlayer(
            Color.RED,
            depth=args.alpha_beta_depth,
            policy_label=f"AlphaBeta(depth={args.alpha_beta_depth})",
        ),
    ]

    baseline = play_report(
        label="baseline",
        game_seed=game_seed,
        catan_map=catan_map,
        players=baseline_players,
        outcomes=args.outcomes,
        max_turns=args.max_turns,
    )
    heuristic = play_report(
        label=args.strategy,
        game_seed=game_seed,
        catan_map=catan_map,
        players=heuristic_players,
        outcomes=args.outcomes,
        max_turns=args.max_turns,
    )
    return baseline, heuristic


def color_name(color: Color | None) -> str:
    return color.value if color is not None else "None"


def print_report(report: GameReport, *, blue_only: bool):
    print(f"  {report.label}:")
    if report.winner is not None or report.truncated:
        suffix = " (max-turn truncation)" if report.truncated else ""
        print(f"    winner={color_name(report.winner)}{suffix}")

    visible_decisions = [
        decision
        for decision in report.decisions
        if not blue_only or decision.color == Color.BLUE
    ]
    for decision in visible_decisions:
        cap = (
            "?"
            if decision.capstone_action is None
            else str(decision.capstone_action)
        )
        print(
            "    "
            f"ply={decision.ply:02d} "
            f"{decision.color.value:<4s} "
            f"{decision.policy:<28s} "
            f"{decision.action_type:<16s} "
            f"cap={cap:<3s} "
            f"{decision.detail}"
        )


def print_summary(pairs: list[tuple[GameReport, GameReport]], *, outcomes: bool):
    if not outcomes:
        print("\nOutcome play disabled. Pass --outcomes to continue games after opening.")
        return

    baseline_blue_wins = sum(
        1 for baseline, _ in pairs if baseline.winner == Color.BLUE
    )
    heuristic_blue_wins = sum(
        1 for _, heuristic in pairs if heuristic.winner == Color.BLUE
    )
    baseline_only = sum(
        1
        for baseline, heuristic in pairs
        if baseline.winner == Color.BLUE and heuristic.winner != Color.BLUE
    )
    heuristic_only = sum(
        1
        for baseline, heuristic in pairs
        if baseline.winner != Color.BLUE and heuristic.winner == Color.BLUE
    )

    n = len(pairs)
    print("\nPaired outcome summary")
    print(f"  baseline blue wins:  {baseline_blue_wins}/{n}")
    print(f"  test blue wins:      {heuristic_blue_wins}/{n}")
    print(f"  baseline-only wins:  {baseline_only}")
    print(f"  test-only wins:      {heuristic_only}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace opening settlements and roads for pure AlphaBeta versus "
            "a placement strategy on one fixed board."
        )
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="heuristic",
        choices=[
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
        help="BLUE placement strategy to compare against AlphaBeta.",
    )
    parser.add_argument(
        "--board-seed",
        type=int,
        default=42,
        help="Fixed board seed. Game seeds vary, but the board stays constant.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First game seed to run.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=3,
        help="Number of consecutive game seeds to inspect.",
    )
    parser.add_argument(
        "--alpha-beta-depth",
        type=int,
        default=2,
        help="Depth for AlphaBeta decisions in both baseline and main-game play.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Heuristic placement sampling temperature. 0 means deterministic argmax.",
    )
    parser.add_argument(
        "--outcomes",
        action="store_true",
        help="After tracing openings, continue each paired game to a winner.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=TURNS_LIMIT,
        help="Turn cap used only with --outcomes.",
    )
    parser.add_argument(
        "--blue-only",
        action="store_true",
        help="Print only BLUE opening decisions instead of both players.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")
    if args.alpha_beta_depth < 0:
        raise SystemExit("--alpha-beta-depth must be non-negative")
    if args.max_turns <= 0:
        raise SystemExit("--max-turns must be positive")

    catan_map = fixed_map(args.board_seed)
    pairs = []
    seed_stop = args.seed_start + args.seed_count

    print(
        "Opening placement comparison "
        f"(strategy={args.strategy}, board_seed={args.board_seed}, "
        f"seeds={args.seed_start}..{seed_stop - 1})"
    )
    print(f"repo={os.fspath(REPO_ROOT)}")

    for game_seed in range(args.seed_start, seed_stop):
        baseline, heuristic = run_pair(args, game_seed, catan_map)
        pairs.append((baseline, heuristic))

        print(f"\nseed={game_seed}")
        print_report(baseline, blue_only=args.blue_only)
        print_report(heuristic, blue_only=args.blue_only)

    print_summary(pairs, outcomes=args.outcomes)


if __name__ == "__main__":
    main()
