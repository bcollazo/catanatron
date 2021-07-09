import json

from catanatron.json import GameEncoder
from catanatron.game import Game
from catanatron.models.player import SimplePlayer, Color
from catanatron_gym.features import create_sample


# Things to benchmark. create_sample(), game.play() (random game), .to_json(), .copy()
def test_to_json_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players)

    result = benchmark(json.dumps, game, cls=GameEncoder)
    assert isinstance(result, str)


def test_copy_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players)

    result = benchmark(game.copy)
    assert result.seed == game.seed


def test_create_sample_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    for _ in range(30):
        game.play_tick()

    sample = benchmark(create_sample, game, players[1].color)
    assert isinstance(sample, dict)
    assert len(sample) > 0
