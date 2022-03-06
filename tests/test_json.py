import json

from catanatron.game import Game
from catanatron.models.enums import ActionType
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


def test_action_from_json():
    data = ["RED", "MARITIME_TRADE", ["SHEEP", "SHEEP", "SHEEP", "SHEEP", "ORE"]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MARITIME_TRADE
    assert action.value == ("SHEEP", "SHEEP", "SHEEP", "SHEEP", "ORE")
