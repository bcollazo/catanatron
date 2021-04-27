from experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    ValueFunctionPlayer,
)
from catanatron.models.player import Color, RandomPlayer
from experimental.play import play_batch
from bayes_opt import BayesianOptimization


def black_box_function(x):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # Needs to use the above params as weights for a players
    players = [
        ValueFunctionPlayer(Color.RED, "Foo", "build_value_function", [4]),
        ValueFunctionPlayer(Color.BLUE, "Bar", "build_value_function", [x]),
    ]
    # players = [
    #     AlphaBetaPlayer(Color.RED, "Foo", "build_value_function", [4]),
    #     AlphaBetaPlayer(Color.BLUE, "Bar", "build_value_function", [x]),
    # ]
    wins, results_by_player = play_batch(100, players, None, False, False)
    vps = results_by_player[players[1].color]
    avg_vps = sum(vps) / len(vps)
    return 100 * wins[str(players[1])] + avg_vps


# Bounded region of parameter space
pbounds = {"x": (0, 10)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)
print(optimizer.max)
