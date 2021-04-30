import timeit

setup = """
import numpy as np
import pickle
from catanatron.game import Game, State
from catanatron.models.player import RandomPlayer, Color
from catanatron.models.decks import ResourceDeck
from catanatron.models.enums import Resource, BuildingType
from catanatron.models.actions import Action, ActionType

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

NUMBER = 1000
result = timeit.timeit("game.copy()", setup=setup, number=NUMBER)
print(result / NUMBER, "secs; game.copy()")

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
state_copy['players_by_color'] = {p.color: p for p in players}
state_copy['board'] = board
state_copy['actions'] = game.state.actions.copy()
state_copy['resource_deck'] = pickle.loads(pickle.dumps(game.state.resource_deck))
state_copy['development_deck'] = pickle.loads(pickle.dumps(game.state.development_deck))
state_copy['tick_queue'] = game.state.tick_queue.copy()
state_copy['current_player_index'] = game.state.current_player_index
state_copy['num_turns'] = game.state.num_turns
state_copy['road_color'] = game.state.road_color
state_copy['army_color'] = game.state.army_color

game_copy = Game([], None, None, initialize=False)
game_copy.seed = game.seed
game_copy.id = game.id
game_copy.state = state_copy
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs; hand-hydrated")

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
'tick_queue': game.state.tick_queue.copy(),
'actions': game.state.actions.copy(),
}
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs; theoretical-limit? (arrays + dicts + map-reuse)")

# Understanding what improvements can we get if we move state to numpy arrays.
result = timeit.timeit(
    """
player = game.state.players[0]
has_money = player.resource_deck.includes(ResourceDeck.city_cost())
has_cities_available = player.cities_available > 0

a = [
    Action(player.color, ActionType.BUILD_CITY, node_id)
    for node_id in player.buildings[BuildingType.SETTLEMENT]
]
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")
result = timeit.timeit(
    """
(player_state[4:9] <= cost).all()
a = np.add(np.where((player_state[4:9] > cost)), 10)
""",
    setup=setup,
    number=NUMBER,
)
print(result / NUMBER, "secs")


# Results:
# 0.0006037695949198678 secs; game.copy()
# 4.2668479960411786e-05 secs; hand-hydrated
# 6.549044977873564e-06 secs; theoretical-limit? (arrays + dicts + map-reuse)
# 9.11637395620346e-06 secs
# 8.073947974480688e-06 secs
