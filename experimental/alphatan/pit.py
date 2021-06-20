from simple_alpha_zero import AlphaTan, create_model, load_replay_memory, pit
from catanatron.models.player import Color, Player, RandomPlayer
from experimental.play import play_batch

model = create_model()
# model.load_weights("data/checkpoints/alphatan")

# For testing...
players = [
    RandomPlayer(Color.ORANGE),
    AlphaTan(Color.WHITE, model, temp=0),
]
wins, vp_history = play_batch(10, players, None, False, False, True)
