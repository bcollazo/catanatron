import copy
import json

from catanatron.models.player import Color, SimplePlayer
from catanatron.json import GameEncoder
from catanatron.game import Game
from catanatron.models.enums import Resource
from catanatron.models.actions import Action, ActionType


def test_play_many_games():
    for _ in range(10):  # play 10 games
        players = [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
        game = Game(players)
        game.play()


def test_copy():
    """Play 30 moves, copy game, ensure they look the same but not the same."""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    for i in range(30):
        game.play_tick()

    game_copy = game.copy()
    assert json.dumps(game, cls=GameEncoder) == json.dumps(game_copy, cls=GameEncoder)
    assert game_copy != game


def test_execute_action_on_copies_doesnt_conflict():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    p0 = game.state.players[0]
    game.execute(Action(p0.color, ActionType.BUILD_FIRST_SETTLEMENT, 0))

    action = Action(p0.color, ActionType.BUILD_INITIAL_ROAD, (0, 1))

    game_copy = game.copy()
    game_copy.execute(action)

    game_copy = game.copy()
    game_copy.execute(action)

    game.execute(action)
