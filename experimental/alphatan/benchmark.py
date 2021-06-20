from catanatron.game import Game

from catanatron.models.player import Color, Player, RandomPlayer
from experimental.alphatan.mcts import AlphaMCTS, game_end_value

from experimental.alphatan.simple_alpha_zero import (
    AlphaTan,
    create_model,
    load_replay_memory,
    pit,
)

players = [RandomPlayer(Color.RED), RandomPlayer(Color.WHITE)]
game = Game(players)


model = create_model()
mcts = AlphaMCTS(model)

for i in range(1000):
    mcts.search(game, Color.RED)
