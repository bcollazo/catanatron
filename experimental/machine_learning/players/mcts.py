import time
import random
import copy
from multiprocessing import Pool
import multiprocessing

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer

DEFAULT_NUM_PLAYOUTS = 25
NUM_WORKERS = 8

# Single threaded NUM_PLAYOUTS=25 takes ~185.3893163204193 secs on initial placement
#   10.498431205749512 secs to do initial road (3 playable actions)
# Multithreaded, dividing the NUM_PLAYOUTS only (actions serially), takes ~52.22048330307007 secs
#   on intial placement. 4.187309980392456 secs on initial road.
# Multithreaded, on different actions
class MCTSPlayer(Player):
    """For each playable action, play N random playouts."""

    def __init__(self, color, name, num_playouts=DEFAULT_NUM_PLAYOUTS):
        super().__init__(color, name=name)
        self.num_playouts = num_playouts

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_action = None
        max_wins = None
        for action in playable_actions:
            params = []
            game_copy = game.copy()
            action_copy = copy.deepcopy(action)
            game_copy.execute(action_copy)
            for _ in range(self.num_playouts):
                params.append((game_copy.copy(), self.color))
            with multiprocessing.Pool(NUM_WORKERS) as p:
                wins = sum(p.map(run_playouts, params))
            if max_wins is None or wins > max_wins:
                best_action = action
                max_wins = wins

        return best_action


def run_playouts(params):
    game_copy, color = params
    game_copy.play(decide_fn=decide_fn)
    winner = game_copy.winning_player()
    return winner is not None and winner.color == color


def decide_fn(self, game, playable_actions):
    index = random.randrange(0, len(playable_actions))
    return playable_actions[index]
