import timeit

setup = """
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from experimental.machine_learning.features import (
    create_sample_vector, expansion_features, reachability_features,
    graph_features
)

game = Game(
    [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
)
game.play()
"""

NUMBER = 1000  # usually a game has around 300 turns and 1000 ticks
result = timeit.timeit(
    "create_sample_vector(game, game.state.players[0].color)",
    setup=setup,
    number=NUMBER,
)
print("create_sample_vector\t", result / NUMBER, "secs")

result = timeit.timeit(
    "expansion_features(game, game.state.players[0].color)",
    setup=setup,
    number=NUMBER,
)
print("expansion_features\t", result / NUMBER, "secs")

result = timeit.timeit(
    "reachability_features(game, game.state.players[0].color)",
    setup=setup,
    number=NUMBER,
)
print("reachability_features\t", result / NUMBER, "secs")

result = timeit.timeit(
    "graph_features(game, game.state.players[0].color)",
    setup=setup,
    number=NUMBER,
)
print("graph_features\t\t", result / NUMBER, "secs")

# Notes:
# road seems to add 0.0025 secs
# production_features don't seem to add much.
# expansion_features seem to add 0.009

# Results:
# create_sample_vector 0.0014413009049603716 secs
# expansion_features 0.0018780140690505505 secs
# reachability_features 0.0006870161320548505 secs
