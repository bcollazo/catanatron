from experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.game import Game
import time
from pprint import pprint
from collections import defaultdict

import tensorflow as tf

from catanatron_server.database import get_last_game_state, save_game_state
from experimental.machine_learning.features import create_sample, get_feature_ordering
from experimental.machine_learning.board_tensor_features import (
    NUMERIC_FEATURES,
    create_board_tensor,
)
from experimental.machine_learning.players.reinforcement import get_t_model, get_v_model

# ===== Read specific underway game
# uuid = "27df8c6b-0d19-4805-a016-9bfbfe5efee6"
# game = get_last_game_state(uuid)

# time1 = time.time()
# PLAYOUTS = 50  # 0.3 secs per game. 50 * 0.3 secs = 15 secs per-move analysis
# wins = defaultdict(int)
# for i in range(PLAYOUTS):
#     game_copy = game.copy()
#     game_copy.play()

#     winner = game_copy.winning_player()
#     winner = None if winner is None else winner.color
#     print(i, winner)
#     wins[winner] += 1

# print(game.num_turns)
# pprint(dict(wins))
# print("took", time.time() - time1)

# # ===== Predict state value
# inputs1 = []
# inputs2 = []
# samples = []
# for player in game.players:
#     sample = create_sample(game, player)
#     state = [float(sample[i]) for i in get_feature_ordering()]
#     samples.append(state)

#     board_tensor = create_board_tensor(game, player)
#     inputs1.append(board_tensor)

#     input2 = [float(sample[i]) for i in get_feature_ordering() if i in NUMERIC_FEATURES]
#     inputs2.append(input2)

# scores = get_v_model("experimental/models/vp-big-256-64").call(
#     tf.convert_to_tensor(samples)
# )
# print(scores)

# scores = get_t_model("experimental/models/tensor-model-normalized").call(
#     [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
# )
# print(scores)
# print(game.players)


# ===== Start a game from scratch. In a specific board. is there better position?
players = [
    MCTSPlayer(Color.RED, "Foo", 10),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.id)
save_game_state(game)

game.play()
