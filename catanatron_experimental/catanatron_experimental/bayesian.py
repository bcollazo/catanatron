from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger

from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    DEFAULT_WEIGHTS,
    ValueFunctionPlayer,
)
from catanatron.models.player import Color
from catanatron_experimental.play import play_batch


def black_box_function(a, b, c, d, e, f, g, h, i, j, k, l):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # Needs to use the above params as weights for a players
    weights = {
        # Where to place. Note winning is best at all costs
        "public_vps": a,
        "production": b,
        "enemy_production": -c,
        "num_tiles": d,
        # Towards where to expand and when
        "reachable_production_0": e,
        "reachable_production_1": f,
        "buildable_nodes": g,
        "longest_road": h,
        # Hand, when to hold and when to use.
        # TODO: Hand synergy
        "hand_resources": i,
        "discard_penalty": -j,
        "hand_devs": k,
        "army_size": l,
    }

    players = [
        # AlphaBetaPlayer(Color.RED, 2, True),
        # AlphaBetaPlayer(Color.BLUE, 2, True, "C", weights),
        ValueFunctionPlayer(Color.RED, "C", params=DEFAULT_WEIGHTS),
        ValueFunctionPlayer(Color.BLUE, "C", params=weights),
    ]
    wins, results_by_player = play_batch(100, players)
    vps = results_by_player[players[1].color]
    avg_vps = sum(vps) / len(vps)
    return 100 * wins[players[1].color] + avg_vps


logger = JSONLogger(path="./bayesian-logs.json")

# Bounded region of parameter space
pbounds = {k: (1e-5, 1e5) for k in "abcdefghijkl"}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=100,
    n_iter=10,
)
print(optimizer.res)
print(optimizer.max)

# weights = DEFAULT_WEIGHTS
# weights[0] = x
# [{'target': 4807.97, 'params': {'x': 417022004708403.75}}, {'target': 5307.94, 'params': {'x': 720324493444954.9}}, {'target': 4307.08, 'params': {'x': 114374827343.74289}}, {'target': 5107.79, 'params': {'x': 302332572638816.44}}, {'target': 4907.91, 'params': {'x': 146755890825645.5}}, {'target': 5408.04, 'params': {'x': 92338594777874.4}}, {'target': 5007.67, 'params': {'x': 186260211385808.3}}, {'target': 4707.85, 'params': {'x': 345560727049592.1}}, {'target': 4707.46, 'params': {'x': 396767474236702.25}}, {'target': 4707.62, 'params': {'x': 538816734007968.75}}, {'target': 4507.39, 'params': {'x': 750295369467801.0}}, {'target': 5007.93, 'params': {'x': 830388995654446.9}}, {'target': 5207.83, 'params': {'x': 982879240298986.4}}, {'target': 4307.21, 'params': {'x': 2168831648142.8152}}, {'target': 4307.45, 'params': {'x': 842315866206777.5}}, {'target': 4307.49, 'params': {'x': 514643074738929.44}}, {'target': 5808.31, 'params': {'x': 34385842392800.824}}, {'target': 4207.53, 'params': {'x': 125932788855395.05}}, {'target': 4407.45, 'params': {'x': 58052731365039.98}}, {'target': 4007.23, 'params': {'x': 928337668753987.0}}]
# {'target': 5808.31, 'params': {'x': 34385842392800.824}}

# Two variables with init_points=10 and n_iter=10 took 30mins
# 3 vars took 60 minutes.
# {'target': 5608.18, 'params': {'x': 140386938603829.9, 'y': 198101489092897.78, 'z': 800744568677529.2}}
