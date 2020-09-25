import os
import json
import uuid
from pathlib import Path

from database import save_game
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
    ]
)
game.play()
print({p.color.value: p.actual_victory_points for p in game.players})

print("Saving game...")
game_id = str(uuid.uuid4())
save_game(game_id, game)

print("See result at http://localhost:3000/games/" + game_id)
