from catanatron import Game, RandomPlayer, Color, GameAccumulator

from catanatron_experimental.my_player import MyPlayer


def test_top_level_imports_work():
    class MyAccumulator(GameAccumulator):
        pass

    players = [
        MyPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.play(accumulators=[MyAccumulator()])
