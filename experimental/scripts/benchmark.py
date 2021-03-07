import pickle
import time
import sys
import numpy as np
import copy
import networkx as nx

from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color

import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


game = Game(
    [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
)
game.play()
print(sys.getsizeof(game))
print(getsize(game))
print(game)

start = time.time()
copy.deepcopy(game)
end = time.time()
print("copy.deepcopy(game) took", end - start, "seconds")

start = time.time()
game.copy()
end = time.time()
print("game.copy() took", end - start, "seconds")

start = time.time()
a = np.random.randint(0, 2, size=(500, 500))
end = time.time()
print("Create Numpy Vector", end - start, "seconds")
print(sys.getsizeof(a))
print(getsize(a))

start = time.time()
np.copy(a)
end = time.time()
print("np.copy(a) took", end - start, "seconds")

start = time.time()
nxgraph = nx.DiGraph(a)
end = time.time()
print("nxgraph = nx.DiGraph(a)", end - start, "seconds")

start = time.time()
game.state.board.nxgraph.copy()
end = time.time()
print("nxgraph.copy()", end - start, "seconds")
