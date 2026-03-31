"""
Generate initial settlement / road phase data and save as pickle files.

Each file is ``pickle.dump([steps, winner], f)`` where:
  - ``steps`` is a list of 8 ``[obs_before, obs_after]`` pairs (2 players)
  - ``winner`` is the winning color's ``.value`` string (e.g. ``"RED"``)

Games without a winner (turn limit) are skipped — ``InitialPhaseFeatureAccumulator``
only records when the game finishes with a winner.

Run from repo root with catanatron installed (``pip install -e catanatron/``) or
``PYTHONPATH`` including ``catanatron/``.

Example:
  python training_data_initial_settlements/generate_initial_settlement_data.py --num-games 50
"""

from __future__ import annotations

import argparse
import os
import pickle

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.gym.initial_phase_accumulator import InitialPhaseFeatureAccumulator
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


def default_players():
    return [
        MCTSPlayer(Color.RED),
        WeightedRandomPlayer(Color.BLUE),
    ]


def run_one_game(players):
    """Run one game; return dict with steps, winner, num_setup_steps, or None."""
    for p in players:
        p.reset_state()
    acc = InitialPhaseFeatureAccumulator()
    game = Game(players)
    game.play(accumulators=[acc])

    if not acc.initial_phase_by_game:
        return None

    steps, winner = acc.initial_phase_by_game[-1]
    return {
        "steps": steps,
        "winner": winner,
        "num_setup_steps": len(steps),
        "game_id": str(game.id),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="How many games to play (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write .pkl files (default: this script's directory)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    players = default_players()
    saved = 0
    skipped = 0

    for _ in range(args.num_games):
        data = run_one_game(players)
        if data is None:
            skipped += 1
            continue
        path = os.path.join(out_dir, f"{data['game_id']}_initial.pkl")
        with open(path, "wb") as f:
            pickle.dump([data["steps"], data["winner"]], f)
        saved += 1
        print(f"Wrote {path}  winner={data['winner']}  steps={data['num_setup_steps']}")

    print(f"Done. Saved {saved} pickle file(s), skipped {skipped} (no winner).")


if __name__ == "__main__":
    main()
"""
#to run this script: 
python3.11 --version
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gym]"
python training_data_initial_settlements/generate_initial_settlement_data.py --num-games 50



"""