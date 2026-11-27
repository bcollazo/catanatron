"""Paired-seat challenge against the Catanatron project's published 1v1 leader."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from itertools import repeat
import json
import multiprocessing
import os
from pathlib import Path
from random import Random
import sys

import catanatron
from catanatron import Color, Game
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionType
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import get_actual_victory_points

from agent import CatanAgent, Weights


ROOT = Path(__file__).resolve().parent
ENGINE_COMMIT = "ecf931181b9a65bb4116a2153fb78c16f1438e00"
ENGINE_SHA = "d17c64ace107eae861082e75fdeda02c8f18b9f029e377881987ab768f89e7b2"
COLORS = (Color.RED, Color.BLUE)


def installed_engine_hash():
    package = Path(catanatron.__file__).parent
    digest = sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def setup_signature(game):
    tiles = [
        (coord, type(tile).__name__, getattr(tile, "resource", None),
         getattr(tile, "number", None), tile.direction.value if hasattr(tile, "direction") else None)
        for coord, tile in sorted(game.state.board.map.tiles.items())
    ]
    data = (tiles, game.state.development_listdeck, game.random.getstate())
    return sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def play_pair(seed, weights):
    order = Random(seed).sample(COLORS, len(COLORS))
    legs = []
    signatures = []
    for seat, color in enumerate(order):
        players = [
            CatanAgent(c, weights, negotiate=False) if c == color else
            AlphaBetaPlayer(c, AlphaBetaPlayer.Params(depth=2))
            for c in COLORS
        ]
        game = Game(players, seed=seed)
        if game.state.colors[seat] != color:
            raise RuntimeError("The seeded referee did not assign the expected seat")
        signatures.append(setup_signature(game))
        winner = game.play()
        offers = sum(a.action_type == ActionType.OFFER_TRADE for a, _ in game.state.action_records)
        if offers:
            raise RuntimeError("Unexpected domestic offer in the stock search benchmark")
        enemy = order[1 - seat]
        legs.append({
            "candidate_seat": seat + 1, "candidate_color": color.value,
            "winner": "candidate" if winner == color else "alphabeta" if winner == enemy else None,
            "candidate_vp": get_actual_victory_points(game.state, color),
            "alphabeta_vp": get_actual_victory_points(game.state, enemy),
            "turns": game.state.num_turns, "domestic_offers": offers,
        })
    if signatures[0] != signatures[1]:
        raise RuntimeError("Paired legs have different initial maps, decks or RNG states")
    return {"seed": seed, "setup_sha256": signatures[0], "legs": legs}


def summarize(pairs):
    legs = [leg for pair in pairs for leg in pair["legs"]]
    scores = [sum(leg["winner"] == "candidate" for leg in pair["legs"]) / 2 for pair in pairs]
    rng = Random(175983107)
    samples = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(10000))
    return {
        "games": len(legs),
        "candidate_wins": sum(leg["winner"] == "candidate" for leg in legs),
        "alphabeta_wins": sum(leg["winner"] == "alphabeta" for leg in legs),
        "unfinished": sum(leg["winner"] is None for leg in legs),
        "candidate_win_rate": sum(scores) / len(scores),
        "candidate_win_rate_95pct_board_bootstrap": [samples[249], samples[9749]],
        "candidate_wins_by_seat": [sum(p["legs"][s]["winner"] == "candidate" for p in pairs) for s in (0, 1)],
        "boards_with_0_1_2_candidate_wins": [scores.count(k / 2) for k in (0, 1, 2)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=64)
    parser.add_argument("--seed", type=int, default=948561723)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights-3p-tactical.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/catanatron-duel.json")
    args = parser.parse_args()
    if args.boards <= 0 or args.workers <= 0:
        parser.error("boards and workers must be positive")
    if installed_engine_hash() != ENGINE_SHA:
        parser.error("install the unmodified Catanatron revision pinned in requirements.txt")
    os.environ["PYTHONHASHSEED"] = "0"  # Spawned workers inherit this before interpreter startup.
    weights = Weights.load(args.weights)
    policy_hash = sha256((ROOT / "agent.py").read_bytes()).hexdigest()
    rng = Random(args.seed)
    seeds = [rng.getrandbits(64) for _ in range(args.boards)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pairs = []
        for pair in pool.map(play_pair, seeds, repeat(weights)):
            pairs.append(pair)
            if len(pairs) % 8 == 0:
                print(f"{len(pairs)}/{args.boards} board pairs complete", flush=True)
    if policy_hash != sha256((ROOT / "agent.py").read_bytes()).hexdigest():
        raise RuntimeError("The policy source changed during the benchmark")
    report = {
        "benchmark": "Catanatron project leaderboard, native two-player challenge",
        "leaderboard": "https://docs.catanatron.com/advanced/making-catanatron-stronger",
        "upstream_commit": ENGINE_COMMIT, "upstream_all_python_sha256": ENGINE_SHA,
        "python_version": sys.version.split()[0], "networkx_version": version("networkx"),
        "python_hash_seed_in_workers": 0, "board_seed_generator": args.seed,
        "boards": args.boards, "workers": args.workers, "policy_file_sha256": policy_hash,
        "policy_weights": asdict(weights),
        "candidate": "CatanAgent with trained three-player weights and domestic proposals disabled only in this duel",
        "opponent": "Unmodified catanatron.players.minimax.AlphaBetaPlayer, Params(depth=2), all other defaults",
        "referee": "Unmodified catanatron.Game.play(), default base map, official_spiral, 10 VP, discard limit 7, friendly robber off",
        "turn_limit": TURNS_LIMIT,
        "pairing": "Seats exchanged on the same starting map, development deck and initial RNG state. Stock search shares the referee RNG, so later dice can differ.",
        "scope": "The project documents a two-player reference comparison. This does not measure the three-player base game or its active bargaining. Stock search cannot evaluate domestic trade replies. Maritime and port trades remain legal.",
        "uncertainty": "95% percentile bootstrap across complete board pairs, 10000 resamples, seed 175983107. Unfinished games count as candidate non-wins.",
        "summary": summarize(pairs), "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
