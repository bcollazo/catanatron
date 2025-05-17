from ray import tune

from catanatron.models.player import Color
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    ValueFunctionPlayer,
)
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.suggestion import ConcurrencyLimiter

from catanatron.cli.play import play_batch


def objective(config):
    players = [
        # AlphaBetaPlayer(Color.RED, 2, True),
        # AlphaBetaPlayer(Color.BLUE, 2, True, "C", weights),
        ValueFunctionPlayer(Color.RED, "C", params=DEFAULT_WEIGHTS),
        ValueFunctionPlayer(Color.BLUE, "C", params=config),
    ]
    wins, results_by_player = play_batch(100, players)
    vps = results_by_player[players[1].color]
    avg_vps = sum(vps) / len(vps)
    score = 100 * wins[players[1].color] + avg_vps

    tune.report(score=score)  # This sends the score to Tune.


def trainable(config):
    # config (dict): A dict of hyperparameters.
    for _ in range(20):
        score = objective(config)

        tune.report(score=score)  # This sends the score to Tune.


analysis = tune.run(
    trainable,
    config={
        # Where to place. Note winning is best at all costs
        "public_vps": tune.uniform(0.0, 100.0),
        "production": tune.uniform(0.0, 100.0),
        "enemy_production": tune.uniform(-100.0, 0.0),
        "num_tiles": tune.uniform(0.0, 100.0),
        # Towards where to expand and when
        "reachable_production_0": tune.uniform(0.0, 100.0),
        "reachable_production_1": tune.uniform(0.0, 100.0),
        "buildable_nodes": tune.uniform(0.0, 100.0),
        "longest_road": tune.uniform(0.0, 100.0),
        # Hand, when to hold and when to use.
        "hand_synergy": tune.uniform(0.0, 100.0),
        "hand_resources": tune.uniform(0.0, 100.0),
        "discard_penalty": tune.uniform(-100.0, 0.0),
        "hand_devs": tune.uniform(0.0, 100.0),
        "army_size": tune.uniform(0.0, 100.0),
    },
    metric="score",
    mode="max",
    # Limit to two concurrent trials (otherwise we end up with random search)
    search_alg=ConcurrencyLimiter(
        BayesOptSearch(random_search_steps=4), max_concurrent=2
    ),
    num_samples=20,
    stop={"training_iteration": 20},
    verbose=2,
)

print("Best config: ", analysis.get_best_config(metric="score", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
breakpoint()
