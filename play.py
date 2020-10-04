import os
import json
import uuid
from pathlib import Path

from database import save_game_state
from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import RandomPlayer, Color

print("Playing game...")
game = Game(
    players=[
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ],
)
game.play()
save_game_state(game)
print({p.color.value: p.actual_victory_points for p in game.players})
print("See result at http://localhost:3000/games/" + game.id)
