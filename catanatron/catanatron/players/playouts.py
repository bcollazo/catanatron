import multiprocessing
import random
import time
from collections import Counter

from catanatron.game import Game
from catanatron.models.player import Player

DEFAULT_NUM_PLAYOUTS = 25
USE_MULTIPROCESSING = True
NUM_WORKERS = multiprocessing.cpu_count()

PLAYOUTS_BUDGET = 100


# Single threaded NUM_PLAYOUTS=25 takes ~185.3893163204193 secs on initial placement
#   10.498431205749512 secs to do initial road (3 playable actions)
# Multithreaded, dividing the NUM_PLAYOUTS only (actions serially), takes ~52.22048330307007 secs
#   on intial placement. 4.187309980392456 secs on initial road.
# Multithreaded, on different actions
class GreedyPlayoutsPlayer(Player):
    """For each playable action, play N random playouts."""

    def __init__(self, color, num_playouts=DEFAULT_NUM_PLAYOUTS):
        super().__init__(color)
        self.num_playouts = int(num_playouts)

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        start = time.time()
        # num_playouts = PLAYOUTS_BUDGET // len(playable_actions)
        num_playouts = self.num_playouts

        best_action = None
        max_wins = None
        for action in playable_actions:
            action_applied_game_copy = game.copy()
            action_applied_game_copy.execute(action)

            counter = run_playouts(action_applied_game_copy, num_playouts)

            wins = counter[self.color]
            if max_wins is None or wins > max_wins:
                best_action = action
                max_wins = wins

        print(
            f"Greedy took {time.time() - start} secs to decide "
            + f"{len(playable_actions)} at {num_playouts} per action"
        )
        return best_action


def run_playouts(action_applied_game_copy, num_playouts):
    start = time.time()
    params = [
        (action_applied_game_copy, seed)
        for seed in _derive_playout_seeds(action_applied_game_copy, num_playouts)
    ]
    if USE_MULTIPROCESSING:
        with multiprocessing.Pool(NUM_WORKERS) as p:
            counter = Counter(p.map(_run_seeded_playout, params))
    else:
        counter = Counter(map(_run_seeded_playout, params))
    duration = time.time() - start
    # print(f"{num_playouts} playouts took: {duration}. Results: {counter}")
    return counter


def _derive_playout_seeds(game, count):
    """Derive distinct deterministic child seeds without advancing the game.

    Pool workers deserialize their inputs independently. Passing the same game
    repeatedly would therefore restart every worker from the same RNG state
    and turn supposedly random playouts into identical simulations.
    """
    seed_rng = random.Random()
    seed_rng.setstate(game.state.random.getstate())
    return [seed_rng.getrandbits(128) for _ in range(count)]


def _run_seeded_playout(params):
    game, seed = params
    game_copy = game.copy()
    rng = random.Random(seed)
    game_copy.random = rng
    game_copy.state.random = rng
    game_copy.play(decide_fn=decide_fn)
    return game_copy.winning_color()


def run_playout(action_applied_game_copy):
    game_copy = action_applied_game_copy.copy()
    game_copy.play(decide_fn=decide_fn)
    return game_copy.winning_color()


def decide_fn(self, game, playable_actions):
    index = game.state.random.randrange(0, len(playable_actions))
    return playable_actions[index]
