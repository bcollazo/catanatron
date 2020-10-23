import json

from catanatron.models.player import Color, SimplePlayer
from catanatron.json import GameEncoder
from catanatron.game import Game, replay_game


def test_play_and_replay_game():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.play()

    replayed = None
    for state in replay_game(game):
        replayed = state

    og_final_state = json.dumps(game, cls=GameEncoder)
    final_state = json.dumps(replayed, cls=GameEncoder)
    assert final_state, og_final_state
