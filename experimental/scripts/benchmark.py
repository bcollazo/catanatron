from catanatron.models.board import STATIC_GRAPH
import pickle
import timeit
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
state = np.random.random((1500,))
end = time.time()
print("Create Numpy Vector", end - start, "seconds")
print(sys.getsizeof(state), getsize(state))

start = time.time()
np.copy(state)
end = time.time()
print("np.copy(a) took", end - start, "seconds")

a = np.random.randint(0, 2, size=(500, 500))
start = time.time()
graph = nx.DiGraph(a)
end = time.time()
print("graph = nx.DiGraph(a)", end - start, "seconds")

start = time.time()
STATIC_GRAPH.copy()
end = time.time()
print("graph.copy()", end - start, "seconds")

# ==== Whats faster to hydrate 4 attributes or 1 numpy array attribute?
NUMBER = 1000
setup = """
import numpy as np

class Container:
    def __init__(self, initialize=True):
        if initialize:
            self.a = 1
            self.b = 3
            self.c = 4
            self.d = 1

class FaseContainer:
    def __init__(self, initialize=True):
        if initialize:
            self.a = np.array([1,3,4,1])

state = Container()
fast = FaseContainer()
"""
result = timeit.timeit(
    """
copy = Container(initialize=False)
copy.a = state.a
copy.b = state.b
copy.c = state.c
copy.d = state.d
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
result = timeit.timeit(
    """
copy = FaseContainer(initialize=False)
copy.a = fast.a.copy()
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
result = timeit.timeit(
    """
{'a': fast.a.copy()}
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
# ==== END. creating new dict of 1 numpy array is fastest. 4 attrs is faster.

# === Its faster to copy dict than numpy array
setup = """
import numpy as np
x = {i: (i, i) for i in range(54)}
x[1] = (3,3)
y = np.zeros((54,2))
"""
result = timeit.timeit(
    "x.copy()",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
result = timeit.timeit(
    "y.copy()",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")

# ===== its faster to .get('1', None)  than try catch
setup = """
x = {i: (i, i) for i in range(54)}
"""
result = timeit.timeit(
    "x.get(80, None)",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
result = timeit.timeit(
    """
try:
    x[80]
except KeyError as e:
    None
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
