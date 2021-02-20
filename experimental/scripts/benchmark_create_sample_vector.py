import timeit

setup = """
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from experimental.machine_learning.features import create_sample_vector

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
    "create_sample_vector(game, game.players[0])", setup=setup, number=NUMBER
)
print(result / NUMBER, "secs")


# Results:
# road seems to add 0.0025 secs
# production_features don't seem to add much.
# expansion_features seem to add 0.009
