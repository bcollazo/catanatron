import json
import uuid

from flask import Flask, jsonify, abort
from flask_cors import CORS

from database import save_game, get_game
from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import RandomPlayer, Color


app = Flask(__name__)
CORS(app)


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
    save_game(game_id, game)
    return jsonify({"game_id": game_id})


@app.route("/games/<string:game_id>", methods=["GET"])
def get_game_endpoint(game_id):
    game = get_game(game_id)
    if game is None:
        abort(404, description="Resource not found")

    return json.dumps(game, cls=GameEncoder)


@app.route("/games/<string:game_id>/tick", methods=["POST"])
def tick_game(game_id):
    game = get_game(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_player() is None:
        game.play_tick()
    save_game(game_id, game)
    return json.dumps(game, cls=GameEncoder)
