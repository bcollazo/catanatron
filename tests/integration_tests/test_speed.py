import json

from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import Color, SimplePlayer, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.features import create_sample
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer

RANDOM_SEED = 0


# Things to benchmark. create_sample(), game.play() (random game), .to_json(), .copy()
def test_to_json_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players, seed=RANDOM_SEED)

    result = benchmark(json.dumps, game, cls=GameEncoder)
    assert isinstance(result, str)


def test_copy_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players, seed=RANDOM_SEED)

    result = benchmark(game.copy)
    assert result.seed == game.seed


def test_create_sample_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players, seed=RANDOM_SEED)
    for _ in range(30):
        game.play_tick()

    sample = benchmark(create_sample, game, players[1].color)
    assert isinstance(sample, dict)
    assert len(sample) > 0


# Benchmarking individual player speeds
def test_simpleplayer_speed(benchmark):
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players, seed=RANDOM_SEED)
    def _play_game(game):
        for _ in range(100):
            game.play_tick()
        return game

    result = benchmark(_play_game, game)


def test_weightedrandom_speed(benchmark):
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        WeightedRandomPlayer(Color.ORANGE),
    ]
    game = Game(players, seed=RANDOM_SEED)
    def _play_game(game):
        for _ in range(100):
            game.play_tick()
        return game

    result = benchmark(_play_game, game)


def test_alphabeta_speed(benchmark):
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        AlphaBetaPlayer(Color.ORANGE),
    ]
    game = Game(players, seed=RANDOM_SEED)
    def _play_game(game):
        for _ in range(100):
            game.play_tick()
        return game

    result = benchmark(_play_game, game)


def test_same_turn_alphabeta_speed(benchmark):
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        SameTurnAlphaBetaPlayer(Color.ORANGE),
    ]
    game = Game(players, seed=RANDOM_SEED)
    def _play_game(game):
        for _ in range(100):
            game.play_tick()
        return game

    result = benchmark(_play_game, game)
