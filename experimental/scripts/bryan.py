from pprint import pprint
import time
import timeit
import numpy as np
import msgpack
import json
from copy import deepcopy
import functools

from experimental.machine_learning.features import create_sample
from catanatron.game import Game
from catanatron.models.player import SimplePlayer, Color
from catanatron.json import GameEncoder


# data = {"name": "John Doe", "ranks": {"sports": 13, "edu": 34, "arts": 45}, "grade": 5}
# array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 1000)
# game = Game(
#     [
#         SimplePlayer(Color.RED),
#         SimplePlayer(Color.BLUE),
#         SimplePlayer(Color.WHITE),
#         SimplePlayer(Color.ORANGE),
#     ],
#     seed=123,
# )
# game.play()
# pprint(game.board.tiles)

# setup = """
# import numpy as np
# import msgpack
# import json
# from copy import deepcopy
# from catanatron.game import Game
# from catanatron.models.player import SimplePlayer, Color
# from catanatron.json import GameEncoder

# data = {'name':'John Doe','ranks':{'sports':13,'edu':34,'arts':45},'grade':5}
# array = np.array([1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1])
# game = Game([SimplePlayer(Color.RED),SimplePlayer(Color.BLUE),SimplePlayer(Color.WHITE),SimplePlayer(Color.ORANGE)])
# """

# # print(timeit.timeit("deepcopy(data)", setup=setup))
# # print(timeit.timeit("json.loads(json.dumps(data))", setup=setup))
# # print(timeit.timeit("msgpack.unpackb(msgpack.packb(data))", setup=setup))
# # print(timeit.timeit("array.copy()", setup=setup))
# # print(timeit.timeit("deepcopy(game)", setup=setup))
# # print(timeit.timeit("json.dumps(game, cls=GameEncoder)", setup=setup))

# x = time.time()
# deepcopy(game)
# print("TOOK", time.time() - x)

# x = time.time()
# json.dumps(game, cls=GameEncoder)
# # json.dumps(game, cls=GameEncoder)
# print("TOOK", time.time() - x)

# x = time.time()
# array.copy()
# array.copy()
# print("TOOK", time.time() - x)

# x = time.time()
# game.copy()
# print("TOOK", time.time() - x)


# @functools.lru_cache(maxsize=None)
# def f(game):
#     print("Computing")
#     return 10


# x = time.time()
# for _ in range(1000):
#     create_sample(game, game.players[0])
# print("create_sample TOOK", time.time() - x)

# breakpoint()

# for _ in range(10):
#     x = time.time()
#     game.copy()
#     game = Game(
#         [
#             SimplePlayer(Color.RED),
#             SimplePlayer(Color.BLUE),
#             SimplePlayer(Color.WHITE),
#             SimplePlayer(Color.ORANGE),
#         ]
#     )
#     print("TOOK", time.time() - x)
