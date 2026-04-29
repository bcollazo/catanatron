Use In Code

From repo root:

import os
import random
import sys

repo = "/Users/peterb/Desktop/academia/catan_ai"
sys.path.insert(0, os.path.join(repo, "capstone_agent"))
sys.path.insert(0, os.path.join(repo, "capstone_agent", "Placement"))

from catanatron.game import Game
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer

from Placement.PlacementAgent import make_placement_agent
from Placement.router_search_player import (
    RouterCapstonePlayer,
    AlphaBetaMainAgentAdapter,
)
from MainPlayAgent import MainPlayAgent


def seed42_board():
    random.seed(42)
    return CatanMap.from_template(BASE_MAP_TEMPLATE)
Champion Placement + AlphaBeta Rest

placement_agent = make_placement_agent("rollout_value_stable")

blue = RouterCapstonePlayer(
    Color.BLUE,
    placement_agent=placement_agent,
    main_agent=AlphaBetaMainAgentAdapter(Color.BLUE, depth=2, prunning=False),
)

red = AlphaBetaPlayer(Color.RED, depth=2)

game = Game(players=[blue, red], catan_map=seed42_board(), seed=750000)
winner = game.play()

print(winner)
For RED instead:

placement_agent = make_placement_agent("rollout_value_stable")

blue = AlphaBetaPlayer(Color.BLUE, depth=2)

red = RouterCapstonePlayer(
    Color.RED,
    placement_agent=placement_agent,
    main_agent=AlphaBetaMainAgentAdapter(Color.RED, depth=2, prunning=False),
)

game = Game(players=[blue, red], catan_map=seed42_board(), seed=750000)
winner = game.play()

print(winner)
Champion Placement + Capstone Rest

placement_agent = make_placement_agent("rollout_value_stable")

main_agent = MainPlayAgent()
main_agent.load("capstone_agent/models/capstone_model.pt")

blue = RouterCapstonePlayer(
    Color.BLUE,
    placement_agent=placement_agent,
    main_agent=main_agent,
)

red = AlphaBetaPlayer(Color.RED, depth=2)

game = Game(players=[blue, red], catan_map=seed42_board(), seed=750000)
winner = game.play()

print(winner)
For RED with Capstone rest:

placement_agent = make_placement_agent("rollout_value_stable")

main_agent = MainPlayAgent()
main_agent.load("capstone_agent/models/capstone_model.pt")

blue = AlphaBetaPlayer(Color.BLUE, depth=2)

red = RouterCapstonePlayer(
    Color.RED,
    placement_agent=placement_agent,
    main_agent=main_agent,
)

game = Game(players=[blue, red], catan_map=seed42_board(), seed=750000)
winner = game.play()

print(winner)
The wrapper doing the routing is router_search_player.py (line 83). It sends placement-phase decisions to rollout_value_stable, and all later decisions to whatever main_agent you pass in.