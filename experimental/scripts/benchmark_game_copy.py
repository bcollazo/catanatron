import timeit

setup = """
import pickle
from catanatron.game import Game, State
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

# board = pickle.loads(pickle.dumps(game.state.board))
board = dict()
board['map'] = game.state.board.map  # for caching speedups
board['nxgraph'] = game.state.board.nxgraph.copy(),
# color => nxgraph.edge_subgraph[] 
# board['connected_components'] = {
#     k: [g.copy() for g in v] 
#     for k, v in game.state.board.connected_components.items()
# }
board['connected_components'] = game.state.board.connected_components.copy()
board['color_node_to_subgraphs'] = pickle.loads(pickle.dumps(game.state.board.color_node_to_subgraphs)),

state_copy = State(None, None, initialize=False)
state_copy.players = players
state_copy.players_by_color = {p.color: p for p in players}
state_copy.board = board
state_copy.actions = game.state.actions.copy()
state_copy.resource_deck = pickle.loads(pickle.dumps(game.state.resource_deck))
state_copy.development_deck = pickle.loads(pickle.dumps(game.state.development_deck))
state_copy.tick_queue = game.state.tick_queue.copy()
state_copy.current_player_index = game.state.current_player_index
state_copy.num_turns = game.state.num_turns
state_copy.road_color = game.state.road_color
state_copy.army_color = game.state.army_color

game_copy = Game([], None, None, initialize=False)
game_copy.seed = game.seed
game_copy.id = game.id
game_copy.state = state_copy
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")


# Results:
# 0.0016250799930421635 secs
# 0.0006950839010532946 secs
