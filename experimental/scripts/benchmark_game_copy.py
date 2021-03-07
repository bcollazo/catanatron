import timeit

setup = """
import pickle
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

NUMBER = 1000
result = timeit.timeit("game.copy()", setup=setup, number=NUMBER)
print(result / NUMBER, "secs")


result = timeit.timeit(
    """
players = pickle.loads(pickle.dumps(game.state.players))
state = {
'a': game.board.nxgraph.copy(),
'b': pickle.loads(pickle.dumps(game.board.connected_components)),
'c': pickle.loads(pickle.dumps(game.board.color_node_to_subgraphs)),
'd': players,
'e': {p.color: p for p in players},
'f': game.state.actions.copy(),
'g': pickle.loads(pickle.dumps(game.state.resource_deck)),
'h': pickle.loads(pickle.dumps(game.state.development_deck)),
'i': game.state.tick_queue,
'j': game.state.current_player_index,
'k': game.state.num_turns,
'l': game.state.road_color,
'm': game.state.army_color
}
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")


# Results:
# 0.0016250799930421635 secs
# 0.0006950839010532946 secs
