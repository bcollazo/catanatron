"""Generate a seeded offline replay using the repository engine."""
import json
import random
from pathlib import Path
from catanatron import Game
from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.json import GameEncoder

random.seed(42)
game = Game([WeightedRandomPlayer(c) for c in Color], seed=42)
states = [json.loads(json.dumps(game, cls=GameEncoder))]
for _ in range(80):
    if game.winning_color():
        break
    game.play_tick()
    states.append(json.loads(json.dumps(game, cls=GameEncoder)))
output = Path(__file__).resolve().parents[1] / "public" / "example-game.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(states, separators=(",", ":")), encoding="utf-8")
(output.parent / "example-preview.json").write_text(json.dumps(states[24], separators=(",", ":")), encoding="utf-8")
print(f"Saved {len(states)} real game snapshots ({output.stat().st_size:,} bytes)")
