#!/usr/bin/env python3
"""Seat-balanced Python player win-rate reference against three Random players."""

from __future__ import annotations

import argparse
import json
import random
import time

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


def make(name: str, color: Color, simulations: int, budget_ms: int):
    if name == "simple": return SimplePlayer(color)
    if name == "random": return RandomPlayer(color)
    if name == "weighted": return WeightedRandomPlayer(color)
    if name == "victory": return VictoryPointPlayer(color)
    if name == "value": return ValueFunctionPlayer(color)
    if name == "playouts": return GreedyPlayoutsPlayer(color, num_playouts=simulations)
    if name == "alphabeta": return AlphaBetaPlayer(color, depth=32)
    if name == "same-turn": return SameTurnAlphaBetaPlayer(color, depth=32)
    if name == "mcts": return MCTSPlayer(color, num_simulations=simulations)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("games", type=int)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--budget-ms", type=int, default=20)
    args = parser.parse_args()
    # AlphaBetaPlayer owns a module-level 20-second deadline; override it only for this matched benchmark.
    import catanatron.players.minimax as minimax
    import catanatron.players.playouts as playouts
    minimax.MAX_SEARCH_TIME_SECS = args.budget_ms / 1000
    playouts.USE_MULTIPROCESSING = False
    wins = truncations = 0
    started = time.perf_counter()
    colors = list(Color)
    for game_index in range(args.games):
        seed = 91 + game_index
        random.seed(10_000 + game_index)
        seat = game_index % 4
        players = [RandomPlayer(color) for color in colors]
        players[seat] = make(args.policy, colors[seat], args.simulations, args.budget_ms)
        game = Game(players, seed=seed)
        game.play()
        winner = game.winning_color()
        wins += winner == colors[seat]
        truncations += winner is None
    seconds = time.perf_counter() - started
    print(json.dumps({
        "engine": "python", "policy": args.policy, "games": args.games,
        "wins": wins, "win_rate": wins / args.games, "truncations": truncations,
        "seconds": seconds, "simulations": args.simulations, "budget_ms": args.budget_ms,
        "seat_rotation": "game_index_mod_4", "opponents": "three Random players"
    }, sort_keys=True))


if __name__ == "__main__":
    main()
