#!/usr/bin/env python3
"""Measure fully completed Python alpha-beta depths under a hard deadline."""

from __future__ import annotations

import argparse
import json
import time

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.tree_search_utils import expand_spectrum
from catanatron.state_functions import player_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("budget_ms", type=int, nargs="?", default=1000)
    parser.add_argument("max_depth", type=int, nargs="?", default=32)
    args = parser.parse_args()
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=91)
    root_color = game.state.current_color()
    def value_fn(position, color):
        return position.state.player_state[
            f"{player_key(position.state, color)}_ACTUAL_VICTORY_POINTS"
        ]
    deadline = time.perf_counter() + args.budget_ms / 1000
    nodes = 0
    completed_depth = 0
    attempted_depth = 0

    def search(position, depth, alpha, beta):
        nonlocal nodes
        nodes += 1
        if time.perf_counter() >= deadline:
            return None, 0.0, False
        if depth == 0 or position.winning_color() is not None:
            return None, value_fn(position, root_color), True
        maximizing = position.state.current_color() == root_color
        best_action = None
        best_value = float("-inf") if maximizing else float("inf")
        for action, outcomes in expand_spectrum(position, position.playable_actions).items():
            expected = 0.0
            for child, probability in outcomes:
                _, value, complete = search(child, depth - 1, alpha, beta)
                if not complete:
                    return best_action, best_value, False
                expected += probability * value
            if (maximizing and expected > best_value) or (
                not maximizing and expected < best_value
            ):
                best_action, best_value = action, expected
            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta:
                break
        return best_action, best_value, True

    started = time.perf_counter()
    for depth in range(1, args.max_depth + 1):
        attempted_depth = depth
        _, _, complete = search(game.copy(), depth, float("-inf"), float("inf"))
        if not complete:
            break
        completed_depth = depth
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(json.dumps({
        "engine": "python",
        "position": "two-player BASE opening, seed 91, official spiral",
        "budget_ms": args.budget_ms,
        "max_depth": args.max_depth,
        "completed_depth": completed_depth,
        "attempted_depth": attempted_depth,
        "nodes": nodes,
        "elapsed_ms": elapsed_ms,
        "heuristic": "actual victory points only",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
