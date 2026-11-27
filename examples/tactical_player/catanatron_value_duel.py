"""Independent 1v1 comparison with the ValueFunctionPlayer discussed in #381."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
from multiprocessing import get_context
import os
from pathlib import Path
from random import Random
import sys

from catanatron import Game
from catanatron.models.enums import ActionType
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points

from agent import CatanAgent, Weights
from catanatron_duel import COLORS, ENGINE_COMMIT, ENGINE_SHA, installed_engine_hash, setup_signature


ROOT = Path(__file__).resolve().parent
SCHEDULE_SEED = 71220674338
POLICY_SHA = "938088b1cff75b2add60e54f06cf12651ef078c0845914e8a88af3bb2f71def2"
WEIGHTS_SHA = "71bab44d172e818e3c40dfb9308186c97bfd4297794ae1667d6fbd0d8b2b1b82"
WEIGHTS = ROOT / "weights-3p-tactical.json"


def play_pair(seed):
    weights = Weights.load(WEIGHTS)
    order = Random(seed).sample(COLORS, len(COLORS))
    legs, signatures = [], []
    for seat, actor in enumerate(order):
        players = [
            CatanAgent(c, weights, negotiate=False) if c == actor else ValueFunctionPlayer(c)
            for c in COLORS
        ]
        game = Game(players, seed=seed)
        if game.state.colors[seat] != actor:
            raise RuntimeError("Unexpected starting seat")
        signatures.append(setup_signature(game))
        winner = game.play()
        if any(r.action.action_type == ActionType.OFFER_TRADE for r in game.state.action_records):
            raise RuntimeError("Unexpected domestic offer")
        legs.append({
            "candidate_seat": seat + 1, "candidate_color": actor.value,
            "winner": winner.value if winner else None,
            "candidate_vp": get_actual_victory_points(game.state, actor),
            "opponent_vp": get_actual_victory_points(game.state, order[1 - seat]),
            "turns": game.state.num_turns,
        })
    if signatures[0] != signatures[1]:
        raise RuntimeError("Paired starting states differ")
    return {"seed": seed, "setup_sha256": signatures[0], "legs": legs}


def summarize(pairs):
    by_seat = [
        sum(p["legs"][seat]["winner"] == p["legs"][seat]["candidate_color"] for p in pairs)
        for seat in (0, 1)
    ]
    per_board = [sum(leg["winner"] == leg["candidate_color"] for leg in p["legs"]) for p in pairs]
    games = 2 * len(pairs)
    rng = Random(175983107)
    bootstrap = sorted(sum(rng.choices(per_board, k=len(per_board))) / games for _ in range(10000))
    unfinished = sum(leg["winner"] is None for p in pairs for leg in p["legs"])
    return {
        "games": games, "candidate_wins": sum(by_seat), "value_wins": games - sum(by_seat) - unfinished,
        "unfinished": unfinished, "wins_by_seat": by_seat, "rate": sum(by_seat) / games,
        "paired_board_bootstrap_95pct": [bootstrap[249], bootstrap[9749]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.boards < 1 or args.workers < 1 or args.output.exists():
        parser.error("Use positive counts and a new report path")
    if (sha256((ROOT / "agent.py").read_bytes()).hexdigest() != POLICY_SHA
            or sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA
            or installed_engine_hash() != ENGINE_SHA):
        parser.error("Expected the unmodified opponent, referee and frozen PR #407 candidate")
    os.environ["PYTHONHASHSEED"] = "0"
    rng = Random(SCHEDULE_SEED)
    seeds = [rng.getrandbits(64) for _ in range(args.boards)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        pairs = list(pool.map(play_pair, seeds, chunksize=2))
    if len({p["setup_sha256"] for p in pairs}) != len(pairs):
        raise ValueError("Unexpected duplicate starting state")
    result = {
        "opponent": "Stock catanatron.players.value.ValueFunctionPlayer() with all default parameters",
        "candidate": "PR #407 frozen tactical player, domestic proposals disabled; bank and port trades available",
        "reason": "Independent comparison with the second documented player discussed in issue #381",
        "source_issue": "https://github.com/bcollazo/catanatron/issues/381",
        "referee": "Stock two-player Game.play(), BASE official_spiral, 10 VP, discard above 7, friendly robber off",
        "source_commit": ENGINE_COMMIT, "engine_sha256": ENGINE_SHA, "policy_sha256": POLICY_SHA, "weights_sha256": WEIGHTS_SHA,
        "python": sys.version.split()[0], "networkx": version("networkx"), "schedule_seed": SCHEDULE_SEED,
        "design": "Same starting board, development deck and RNG for each seat pair. Later draws can diverge. Percentile interval resamples whole boards 10000 times, seed 175983107.",
        "summary": summarize(pairs), "pairs": pairs,
    }
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
