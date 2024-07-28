import pytest
import json

from catanatron.game import Game
from catanatron.models.enums import ActionType, WOOD, BRICK, SHEEP, ORE
from catanatron.models.player import SimplePlayer, Color
from catanatron.json import GameEncoder, action_from_json


def test_serialization():
    game = Game(
        players=[
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
    )

    string = json.dumps(game, cls=GameEncoder)
    result = json.loads(string)

    # Loosely assert looks like expected
    assert isinstance(result["robber_coordinate"], list)
    assert isinstance(result["tiles"], list)
    assert isinstance(result["edges"], list)
    assert isinstance(result["nodes"], dict)
    assert isinstance(result["actions"], list)


def test_action_from_json_maritime_trade():
    data = ["RED", "MARITIME_TRADE", [SHEEP, SHEEP, SHEEP, SHEEP, ORE]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MARITIME_TRADE
    assert action.value == (SHEEP, SHEEP, SHEEP, SHEEP, ORE)


def test_action_from_json_play_year_of_plenty_two_resources():
    data = ["RED", "PLAY_YEAR_OF_PLENTY", [WOOD, BRICK]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.PLAY_YEAR_OF_PLENTY
    assert action.value == (WOOD, BRICK)


def test_action_from_json_play_year_of_plenty_one_resource():
    data = ["BLUE", "PLAY_YEAR_OF_PLENTY", [SHEEP]]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.PLAY_YEAR_OF_PLENTY
    assert action.value == (SHEEP,)


def test_action_from_json_play_year_of_plenty_invalid():
    data = ["WHITE", "PLAY_YEAR_OF_PLENTY", [WOOD, BRICK, SHEEP]]
    with pytest.raises(
        ValueError, match="Year of Plenty action must have 1 or 2 resources"
    ):
        action_from_json(data)


def test_action_from_json_move_robber_with_victim():
    data = ["ORANGE", "MOVE_ROBBER", [[0, 0, 0], "RED", None]]
    action = action_from_json(data)
    assert action.color == Color.ORANGE
    assert action.action_type == ActionType.MOVE_ROBBER
    assert action.value == ((0, 0, 0), Color.RED, None)


def test_action_from_json_move_robber_without_victim():
    data = ["RED", "MOVE_ROBBER", [[1, -1, 0], None, None]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MOVE_ROBBER
    assert action.value == ((1, -1, 0), None, None)


def test_action_from_json_build_road():
    data = ["BLUE", "BUILD_ROAD", [0, 1]]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.BUILD_ROAD
    assert action.value == (0, 1)
