import timeit

setup = """
import numpy as np
import pickle
from catanatron.game import Game, State
from catanatron.models.player import RandomPlayer, Color
from catanatron.models.enums import CITY, SETTLEMENT, Action, ActionType

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
board = dict()
board['map'] = game.state.board.map  # for caching speedups
board['buildings'] = game.state.board.buildings.copy()
board['roads'] = game.state.board.roads.copy()
board['connected_components'] = game.state.board.connected_components.copy()

state_copy = dict()
state_copy['colors'] = game.state.colors.copy()
state_copy['board'] = board
state_copy['action_records'] = game.state.action_records.copy()
state_copy['resource_freqdeck'] = game.state.resource_freqdeck.copy()
state_copy['development_listdeck'] = game.state.development_listdeck.copy()
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
# 3.558104199999998e-05 secs; game.copy()
# 5.459833999999997e-06 secs; hand-hydrated
# 2.3131659999999776e-06 secs; theoretical-limit? (arrays + dicts + map-reuse)
