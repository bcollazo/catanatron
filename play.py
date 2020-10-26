import traceback
import os
import json
import uuid
from pathlib import Path
import random
from collections import defaultdict

import click
import termplotlib as tpl
import numpy

from database import save_game_state
from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import RandomPlayer, Color, SimplePlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


@click.command()
@click.option("--num", default=5, help="Number of greetings.")
def simulate(num):
    """Simple program simulates NUM Catan games."""
    player_classes = [WeightedRandomPlayer, RandomPlayer, RandomPlayer, SimplePlayer]
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    pseudonyms = ["Foo", "Bar", "Baz", "Qux"]

    wins = defaultdict(int)
    for x in range(num):
        seating = random.sample(range(4), 4)
        players = [player_classes[i](colors[i], pseudonyms[i]) for i in seating]

        print("Playing game:", players)
        game = Game(players)
        try:
            game.play()
        except Exception as e:
            traceback.print_exc()
        finally:
            save_game_state(game)
        print({str(p): p.actual_victory_points for p in players})
        print("See result at http://localhost:3000/games/" + game.id)

        winner = game.winning_player()
        wins[str(winner)] += 1

    seating = range(4)
    players = [player_classes[i](colors[i], pseudonyms[i]) for i in seating]
    fig = tpl.figure()
    fig.barh([wins[str(p)] for p in players], players, force_ascii=False)
    fig.show()


if __name__ == "__main__":
    simulate()
