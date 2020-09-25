import json
import uuid

from flask import Flask, jsonify, abort
from flask_cors import CORS

from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import RandomPlayer, Color


app = Flask(__name__)
CORS(app)


games = {}


@app.route("/games", methods=["POST"])
def create_game():
    game = Game(
        players=[
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
    )
    game.play_initial_build_phase()
    game_id = uuid.uuid4()
    games[str(game_id)] = game
    return jsonify({"game_id": game_id})


@app.route("/games/<string:game_id>", methods=["GET"])
def get_game(game_id):
    if game_id not in games:
        abort(404, description="Resource not found")

    game = games[game_id]
    return json.dumps(game, cls=GameEncoder)


@app.route("/games/<string:game_id>/tick", methods=["POST"])
def tick_game(game_id):
    if game_id not in games:
        abort(404, description="Resource not found")

    game = games[game_id]
    game.play_tick()
    return json.dumps(game, cls=GameEncoder)


@app.route("/test", methods=["GET"])
def test():
    game = Game(
        players=[
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
    )
    game.play_initial_build_phase()

    return json.dumps(game, cls=GameEncoder)
