import timeit

setup = """
import numpy as np
import pickle
from catanatron.game import Game, State
from catanatron.models.player import RandomPlayer, Color
from catanatron.models.decks import ResourceDeck
from catanatron.models.enums import Resource, BuildingType, Action, ActionType

game = Game(
    [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
)
game.play()
array_state = np.random.random((14,))
player_state = np.random.random((28 * 4,))
cost = np.array([0,0,2,3,0])
action_space = np.array([i for i in range(5000)])
"""

# 1 =====
NUMBER = 1000
result = timeit.timeit("game.copy()", setup=setup, number=NUMBER)
print(result / NUMBER, "secs; game.copy()")

# 2 =====
# Next step
result = timeit.timeit(
    """
players = game.state.players.copy()

board = dict()
board['map'] = game.state.board.map  # for caching speedups
board['buildings'] = game.state.board.buildings.copy()
board['roads'] = game.state.board.roads.copy()
board['connected_components'] = game.state.board.connected_components.copy()

state_copy = dict()
state_copy['players'] = players
state_copy['board'] = board
state_copy['actions'] = game.state.actions.copy()
state_copy['resource_deck'] = pickle.loads(pickle.dumps(game.state.resource_deck))
state_copy['development_deck'] = pickle.loads(pickle.dumps(game.state.development_deck))
state_copy['current_player_index'] = game.state.current_player_index
state_copy['num_turns'] = game.state.num_turns

game_copy = Game([], None, None, initialize=False)
game_copy.seed = game.seed
game_copy.id = game.id
game_copy.state = state_copy
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs; hand-hydrated")

# 3 =====
# Theoretical Python limit on state.copy(?) (Using numpy arrays)
result = timeit.timeit(
    """
# Players are 24-length array of numbers.
players = player_state.copy()

# Graph is tensor board(?)
board = {
    'map': game.state.board.map,
    'buildings': game.state.board.buildings.copy(),
    'roads': game.state.board.roads.copy(),
    'connected_components': game.state.board.connected_components.copy(),
}

state_copy = {
'players': players,
'board': board,
'array_state': array_state.copy(),
'actions': game.state.actions.copy(),
}
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs; theoretical-limit? (arrays + dicts + map-reuse)")


# Results:
# 0.00045712706199992683 secs; game.copy()
# 8.875908100162633e-05 secs; hand-hydrated
# 1.0377163001976441e-05 secs; theoretical-limit? (arrays + dicts + map-reuse)
# 8.97490599891171e-06 secs
# 8.097766003629659e-06 secs
