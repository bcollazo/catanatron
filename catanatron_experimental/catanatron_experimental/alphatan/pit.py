import uuid

from catanatron_experimental.alphatan.simple_alpha_zero import (
    AlphaTan,
    create_model,
    load_replay_memory,
    pit,
)
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron_experimental.play import play_batch

save_in_db = False
model = create_model()
# model.load_weights("data/checkpoints/alphatan")

players = [
    RandomPlayer(Color.BLUE),
    # AlphaTan(Color.BLUE, uuid.uuid4(), model, temp=0, num_simulations=10),
    AlphaTan(Color.RED, uuid.uuid4(), model, temp=0, num_simulations=10),
]
wins, vp_history = play_batch(10, players)
