import time
import sys
import numpy as np
import copy
import networkx as nx

from catanatron_server.database import get_last_game_state

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


# game_id = "2fd02825-1abb-48dd-953d-a385ee601c45"
game_id = "d9d8e4a8-232d-4f55-b4d3-8dccf6ee84ef"
game = get_last_game_state(game_id)
print(sys.getsizeof(game))
print(getsize(game))
print(game)

start = time.time()
copy.deepcopy(game)
end = time.time()
print("copy.deepcopy(game) took", end - start, "seconds")


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
D = nx.DiGraph(a)
end = time.time()
print("Create Graph from Vector", end - start, "seconds")


D = nx.to_networkx_graph(a, create_using=nx.Graph)
print(D)

out = nx.to_numpy_array(D)
print(out)

out = nx.to_numpy_matrix(D)
print(out)

G = nx.Graph()


import pandas as pd

ids = [11, 22, 33, 44, 55, 66, 77]
countries = ["Spain", "France", "Spain", "Germany", "France"]

df = pd.DataFrame(list(zip(ids, countries)), columns=["Ids", "Countries"])
print(df)
print(df.dtypes)

G = nx.Graph()
G.add_node("C")
G.add_edge("A", "B", weight=4, color="red")
print(G)
print(G.nodes)
print(G.edges)
print(G.nodes.data())
print(G.edges.data())
print(nx.to_numpy_array(G))
UG = nx.to_directed(G)
print(UG)
print(UG.nodes)
print(UG.edges)
print(UG.nodes.data())
print(UG.edges.data())
print(nx.to_numpy_array(UG))
print(nx.dag_longest_path(UG))
