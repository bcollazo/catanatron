import json

from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.json import GameEncoder
from catanatron.game import Game
from catanatron.models.enums import VICTORY_POINT, Action, ActionType, CITY, SETTLEMENT
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_player_buildings,
)


def test_play_many_games():
    for _ in range(10):  # play 10 games
        players = [
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
        game = Game(players)
        game.play()

        # Assert everything looks good
        for color in game.state.colors:
            cities = len(get_player_buildings(game.state, color, CITY))
            settlements = len(get_player_buildings(game.state, color, SETTLEMENT))
            longest = get_longest_road_color(game.state) == color
            largest = get_largest_army(game.state)[0] == color
            devvps = get_dev_cards_in_hand(game.state, color, VICTORY_POINT)
            assert (
                settlements + 2 * cities + 2 * longest + 2 * largest + devvps
            ) == get_actual_victory_points(game.state, color)


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
    p0_color = game.state.colors[0]
    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, 0))

    action = Action(p0_color, ActionType.BUILD_ROAD, (0, 1))

    game_copy = game.copy()
    game_copy.execute(action)

    game_copy = game.copy()
    game_copy.execute(action)

    game.execute(action)
