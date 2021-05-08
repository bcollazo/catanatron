import timeit

setup = """
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color


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
    "game.state.board.buildable_edges(Color.RED)",
    setup=setup,
    number=NUMBER,
)
print("buildable_edges", result / NUMBER, "secs")
