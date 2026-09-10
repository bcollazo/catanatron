#!/usr/bin/env python3
"""Matched CPython full-game baseline for the Rust E08 scoreboard."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    samples: list[float] = []
    ticks = completed = truncated = 0
    random.seed(args.seed)
    fixed_root = Game(
        [RandomPlayer(color) for color in list(Color)[:4]],
        seed=args.seed,
        catan_map=build_map("BASE"),
    )
    for batch in range(5):
        started = time.perf_counter()
        for index in range(args.games):
            seed = args.seed + batch * args.games + index
            random.seed(seed)
            if args.fixed:
                random.seed(seed)
                game = fixed_root.copy()
            else:
                random.seed(seed)
                game = Game(
                    [RandomPlayer(color) for color in list(Color)[:4]],
                    seed=seed,
                    catan_map=build_map("BASE"),
                )
            before = len(game.state.action_records)
            winner = game.play()
            ticks += len(game.state.action_records) - before
            if winner is None:
                truncated += 1
            else:
                completed += 1
        samples.append(time.perf_counter() - started)
    report = {
        "engine": "python",
        "workload": "fixed-rollouts" if args.fixed else "games",
        "rules_profile": "pinned-python",
        "map": "BASE",
        "policy": "random",
        "seed": args.seed,
        "players": 4,
        "games": args.games * 5,
        "completed": completed,
        "truncated": truncated,
        "player_intents": ticks,
        "sample_seconds": samples,
        "intents_per_second": ticks / sum(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
