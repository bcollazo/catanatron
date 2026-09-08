#!/usr/bin/env python3
"""Measure E10 stdio decision latency and fixed-schedule head-to-head results."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
import time
from pathlib import Path

from catanatron import Color, Game
from catanatron.models.player import RandomPlayer
from catanatron.players.stdio import build_stdio_player_class
from catanatron.players.weighted_random import WeightedRandomPlayer


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=Path, required=True)
    parser.add_argument("--games-per-opponent", type=int, default=20)
    parser.add_argument("--budget-ms", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    command = [
        str(args.bot.resolve()), "--policy", "rollout", "--simulations",
        str(args.simulations), "--budget-ms", str(args.budget_ms), "--seed",
        str(args.seed), "--threads", "1",
        "--metrics", "true",
    ]
    base = build_stdio_player_class("MeasuredRustBot", command)
    latencies: list[float] = []

    class MeasuredBot(base):
        def decide(self, game, playable_actions):
            started = time.perf_counter()
            result = super().decide(game, playable_actions)
            latencies.append((time.perf_counter() - started) * 1000)
            return result

    results = {}
    metrics_file = tempfile.TemporaryFile(mode="w+")
    original_stderr = __import__("os").dup(2)
    __import__("os").dup2(metrics_file.fileno(), 2)
    try:
      for label, opponent in (("random", RandomPlayer), ("weighted", WeightedRandomPlayer)):
        wins = losses = draws = 0
        for game_index in range(args.games_per_opponent):
            bot_color = Color.RED if game_index % 2 == 0 else Color.BLUE
            other_color = Color.BLUE if bot_color == Color.RED else Color.RED
            bot = MeasuredBot(bot_color)
            players = [bot, opponent(other_color)] if bot_color == Color.RED else [opponent(other_color), bot]
            seed = args.seed + game_index
            random.seed(seed)
            winner = Game(players, seed=seed).play()
            bot.close()
            if winner == bot_color:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1
        total = wins + losses + draws
        rate = wins / total
        error = 1.96 * math.sqrt(rate * (1 - rate) / total) if total else 0
        results[label] = {
            "games": total, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": rate, "approx_95pct_interval": [max(0, rate-error), min(1, rate+error)],
        }
    finally:
        __import__("os").dup2(original_stderr, 2)
        __import__("os").close(original_stderr)
    metrics_file.seek(0)
    rollout_counts = []
    for line in metrics_file:
        if line.startswith("catanatron_search_metrics "):
            rollout_counts.append(json.loads(line.split(" ", 1)[1])["rollouts"])
    metrics_file.close()

    report = {
        "policy": "rollout", "threads": 1, "budget_ms": args.budget_ms,
        "simulation_cap": args.simulations, "seed_schedule_start": args.seed,
        "decisions": len(latencies),
        "searched_decisions": len(rollout_counts),
        "rollouts_per_searched_decision": {
            "mean": statistics.mean(rollout_counts),
            "median": statistics.median(rollout_counts),
            "min": min(rollout_counts), "max": max(rollout_counts),
        },
        "latency_ms": {
            "p50": statistics.median(latencies), "p95": percentile(latencies, .95),
            "p99": percentile(latencies, .99), "max": max(latencies),
        },
        "head_to_head": results,
        "note": "Intervals are approximate; this sample reports observations, not strength superiority.",
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
