from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger

from experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    DEFAULT_WEIGHTS,
)
from catanatron.models.player import Color
from experimental.play import play_batch


def black_box_function(x):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # Needs to use the above params as weights for a players
    weights = DEFAULT_WEIGHTS
    weights[0] = x
    players = [
        AlphaBetaPlayer(Color.RED, "Foo", 2),
        AlphaBetaPlayer(Color.BLUE, "Bar", 2, "C", weights),
    ]
    wins, results_by_player = play_batch(
        100, players, None, False, False, verbose=False
    )
    vps = results_by_player[players[1].color]
    avg_vps = sum(vps) / len(vps)
    return 100 * wins[str(players[1])] + avg_vps


logger = JSONLogger(path="./bayesian-logs.json")

# Bounded region of parameter space
pbounds = {"x": (1e4, 1e15)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=10,
    n_iter=10,
)
print(optimizer.res)
print(optimizer.max)
