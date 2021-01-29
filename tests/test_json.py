import json

from catanatron.game import Game
from catanatron.models.player import SimplePlayer, Color
from catanatron.json import GameEncoder


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
    assert len(result["players"]) == 4
    assert isinstance(result["robber_coordinate"], list)
    assert isinstance(result["tiles"], list)
    assert isinstance(result["edges"], list)
    assert isinstance(result["nodes"], dict)
    assert isinstance(result["actions"], list)
