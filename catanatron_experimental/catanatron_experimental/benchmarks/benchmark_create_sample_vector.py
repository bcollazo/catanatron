import timeit

setup = """
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron.features import (
    create_sample_vector, expansion_features, reachability_features,
    graph_features, tile_features
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
    "create_sample_vector(game, game.state.colors[0])",
    setup=setup,
    number=NUMBER,
)
print("create_sample_vector\t", result / NUMBER, "secs")

result = timeit.timeit(
    "expansion_features(game, game.state.colors[0])",
    setup=setup,
    number=NUMBER,
)
print("expansion_features\t", result / NUMBER, "secs")

result = timeit.timeit(
    "reachability_features(game, game.state.colors[0])",
    setup=setup,
    number=NUMBER,
)
print("reachability_features\t", result / NUMBER, "secs")

result = timeit.timeit(
    "graph_features(game, game.state.colors[0])",
    setup=setup,
    number=NUMBER,
)
print("graph_features\t\t", result / NUMBER, "secs")


result = timeit.timeit(
    "tile_features(game, game.state.colors[0])",
    setup=setup,
    number=NUMBER,
)
print("tile_features\t\t", result / NUMBER, "secs")

# Notes:
# road seems to add 0.0025 secs
# production_features don't seem to add much.
# expansion_features seem to add 0.009

# Results:
# create_sample_vector	 0.0002994296670076437 secs
# expansion_features	 0.0009035729159950278 secs
# reachability_features	 0.00039436871399811934 secs
# graph_features		 1.5904047002550214e-05 secs
# tile_features		     3.960479953093454e-07 secs
