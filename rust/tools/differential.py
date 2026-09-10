#!/usr/bin/env python3
"""Drive Python games through the Rust JSONL conformance boundary."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import random
import subprocess
import sys
from pathlib import Path

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer

from export_fixtures import action_value, menu, normalize, snapshot

RUST = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = RUST / "target" / "release" / "catanatron-conformance.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", "--games-per-config", dest="games", type=int, default=100)
    parser.add_argument("--fixtures", type=Path, default=RUST / "tests" / "fixtures")
    parser.add_argument("--game-offset", type=int, default=0)
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument("--map", default="BASE", choices=["BASE", "TOURNAMENT", "MINI"])
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games < 1 or any(players not in (2, 3, 4) for players in args.players):
        raise SystemExit("--games must be positive and --players must contain only 2, 3, or 4")
    if not (args.fixtures / "manifest.json").is_file():
        raise SystemExit(f"fixture manifest not found under {args.fixtures}")
    process = subprocess.Popen(
        [str(args.runner), "--allow-known-divergences"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    equal_games = truncated = transitions = 0
    recent: OrderedDict[str, dict] = OrderedDict()
    try:
        for players in args.players:
            for game_index in range(args.game_offset, args.game_offset + args.games):
                seed = args.seed + players * 100_000 + game_index
                random.seed(seed)
                game = Game(
                    [RandomPlayer(color) for color in list(Color)[:players]],
                    seed=seed,
                    catan_map=build_map(args.map),
                )
                selector = random.Random(seed ^ 0xC47A_7A0)
                step = 0
                while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
                    before = snapshot(game, args.map)
                    legal_before = menu(game.playable_actions)
                    ordered = sorted(
                        game.playable_actions,
                        key=lambda item: json.dumps(action_value(item), sort_keys=True),
                    )
                    action = selector.choice(ordered)
                    result = game.execute(action)
                    winner = game.winning_color()
                    record = {
                        "fixture_version": 2,
                        "case_id": f"live-{players}p-{game_index:03d}-{step:05d}",
                        "source_revision": "live-python",
                        "rules_profile": "rust-v1",
                        "before": before,
                        "actor": action.color.value,
                        "action": action_value(action, intent=True),
                        "outcome": normalize(result.result) if result.result is not None else None,
                        "after": snapshot(game, args.map),
                        "legal_before": legal_before,
                        "legal_after": [] if winner is not None else menu(game.playable_actions),
                        "status_after": "won" if winner is not None else "decision",
                    }
                    process.stdin.write(json.dumps(record, separators=(",", ":")) + "\n")
                    recent[record["case_id"]] = record
                    if len(recent) > 2048:
                        recent.popitem(last=False)
                    transitions += 1
                    step += 1
                if game.winning_color() is None:
                    truncated += 1
                else:
                    equal_games += 1
        process.stdin.close()
        stdout = process.stdout.read() if process.stdout is not None else ""
        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
    except BrokenPipeError:
        stdout = process.stdout.read() if process.stdout is not None else ""
        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
    try:
        runner_report = json.loads(stdout)
    except json.JSONDecodeError:
        runner_report = {}
    divergent_games = int(runner_report.get("divergent_games", 0))
    report = {
        "rules_profile": "rust-v1",
        "map": args.map,
        "seed": args.seed,
        "players": args.players,
        "games_per_configuration": args.games,
        "equal_games": equal_games - divergent_games,
        "divergent_games": divergent_games,
        "divergent_transitions": int(runner_report.get("divergent", 0)),
        "truncated_games": truncated,
        "failed_games": 0 if return_code == 0 else 1,
        "transitions": transitions,
        "runner": stdout.strip(),
        "fixtures": str(args.fixtures),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(json.dumps(report, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if return_code != 0:
        print(stderr.strip(), file=sys.stderr)
        if args.failure_output:
            try:
                case_id = json.loads(stderr.strip())["case_id"]
                failed = recent[case_id]
            except (json.JSONDecodeError, KeyError):
                pass
            else:
                args.failure_output.parent.mkdir(parents=True, exist_ok=True)
                args.failure_output.write_text(
                    json.dumps(failed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
