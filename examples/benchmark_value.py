"""Fresh-process full-game check; compare JSON from base and candidate checkouts.

Run with PYTHONHASHSEED=0. --audit checks every scalar against the frozen
pre-optimization evaluator and records leaf order and search deadline cutoffs.
Use disjoint seeds for clean timing and audits. This uses only Catanatron.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players import minimax, value
from catanatron.state_functions import player_key

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--audit", action="store_true")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists")
leaves, deadlines = 0, 0
leaf_digest = hashlib.sha256()
if args.audit:
    reference = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests/value_reference.py")
    )["reference_base_fn"]
    original = value.base_fn

    def factory(params=value.DEFAULT_WEIGHTS, **kwargs):
        fast, native = original(params, **kwargs), reference(params)

        def evaluate(game, color):
            global leaves
            actual = fast(game, color)
            expected = native(game, color)
            assert actual == expected, (actual, expected)
            leaves += 1
            leaf_digest.update(actual.hex().encode() + b"\0")
            return actual

        return evaluate

    value.base_fn = factory
    original_alpha = minimax.AlphaBetaPlayer.alphabeta

    def alpha(self, game, depth, low, high, deadline, node):
        global deadlines
        if depth > 0 and game.winning_color() is None and time.time() >= deadline:
            deadlines += 1
        return original_alpha(self, game, depth, low, high, deadline, node)

    minimax.AlphaBetaPlayer.alphabeta = alpha
players = [value.ValueFunctionPlayer(Color.RED)] + [
    minimax.AlphaBetaPlayer(c) for c in list(Color)[1:]
]
wall, cpu = time.perf_counter(), time.process_time()
game = Game(players, seed=args.seed)
winner = game.play()
seconds, cpu_seconds = time.perf_counter() - wall, time.process_time() - cpu
assert winner is not None
import catanatron

source = hashlib.sha256()
root = Path(catanatron.__file__).parent
for path in sorted(root.rglob("*.py")):
    source.update(
        path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
    )
r = {
    "seed": args.seed,
    "audit": args.audit,
    "seconds": seconds,
    "cpu_seconds": cpu_seconds,
    "python": sys.version,
    "source_sha256": source.hexdigest(),
    "winner": winner.value,
    "points": {
        c.value: game.state.player_state[
            player_key(game.state, c) + "_ACTUAL_VICTORY_POINTS"
        ]
        for c in Color
    },
    "actions": len(game.state.action_records),
    "action_sha256": hashlib.sha256(
        json.dumps([repr(x) for x in game.state.action_records]).encode()
    ).hexdigest(),
    "leaves": leaves if args.audit else None,
    "leaf_sha256": leaf_digest.hexdigest() if args.audit else None,
    "deadline_observations": deadlines if args.audit else None,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
print(json.dumps(r), flush=True)
assert not deadlines, "deadline-limited audit cannot establish equivalence"
