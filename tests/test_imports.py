from catanatron import Game, RandomPlayer, Color, GameAccumulator

from examples.custom_player import FooPlayer


def test_top_level_imports_work():
    class MyAccumulator(GameAccumulator):
        pass

    players = [
        FooPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.play(accumulators=[MyAccumulator()])
