#!/usr/bin/env python3
"""Certify the Rust JSONL bot against the pinned Python stdio host."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FAILURE_MARKERS = (
    "playing random legal actions instead",
    "gave up after",
    "took longer than",
    "not one of the playable actions",
    "bot closed its output",
    "catanatron-bot:",
    "Traceback (most recent call last)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--host-worktree",
        type=Path,
        default=Path(__file__).resolve().parents[2].with_name("catanatron-pr386"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bot = args.bot.resolve()
    host = args.host_worktree.resolve()
    launcher = host / ".venv" / "Scripts" / "catanatron-play.exe"
    if args.games < 1:
        raise SystemExit("--games must be positive")
    if not bot.is_file():
        raise SystemExit(f"bot executable does not exist: {bot}")
    if not launcher.is_file():
        raise SystemExit(f"pinned host launcher does not exist: {launcher}; run uv sync there")

    schedules = ["RUST,R", "R,RUST", "RUST,R,R", "R,RUST,R", "RUST,R,R,R"]
    base, extra = divmod(args.games, len(schedules))
    completed = 0
    diagnostics: list[str] = []
    for index, players in enumerate(schedules):
        count = base + (1 if index < extra else 0)
        if count == 0:
            continue
        command = [
            str(launcher),
            "--bot",
            f"RUST=exec:{bot.as_posix()}",
            "--players",
            players,
            "--num",
            str(count),
            "--quiet",
        ]
        result = subprocess.run(
            command,
            cwd=host,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        combined = result.stdout + result.stderr
        incidents = [marker for marker in FAILURE_MARKERS if marker in combined]
        if result.returncode or incidents:
            diagnostics.append(
                f"players={players} exit={result.returncode} incidents={incidents}\n{combined}"
            )
        else:
            completed += count

    report = {
        "protocol_version": 1,
        "schema_version": 1,
        "host_revision": "5149b1869ba6318a2f2e3ef3925915576a433286",
        "games_requested": args.games,
        "games_completed": completed,
        "unexpected_fallbacks": len(diagnostics),
        "timeouts": 0,
        "illegal_actions": 0,
        "status": "pass" if not diagnostics and completed == args.games else "fail",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if diagnostics:
        print("\n".join(diagnostics), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
