import json

from flask import Flask, jsonify, abort
from flask_cors import CORS
import tensorflow as tf

from catanatron_server.database import save_game_state, get_last_game_state
from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import RandomPlayer, Color
from experimental.machine_learning.players.mcts import MCTSPlayer
from experimental.machine_learning.features import create_sample, get_feature_ordering
from experimental.machine_learning.players.reinforcement import (
    TensorRLPlayer,
    VRLPlayer,
    get_v_model,
)


app = Flask(__name__)
CORS(app)


@app.route("/games/<string:game_id>/tick", methods=["POST"])
def tick_game(game_id):
    game = get_last_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_player() is None:
        game.play_tick(lambda g: save_game_state(g))
    return json.dumps(game, cls=GameEncoder)


@app.route(
    "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
)
def get_game_feature_vector(game_id, player_index):
    game = get_last_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    return create_sample(game, game.players[player_index])


@app.route("/games/<string:game_id>/value-function", methods=["GET"])
def get_game_value_function(game_id):
    game = get_last_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    data = {}
    for player_index in range(4):
        sample = create_sample(game, game.players[player_index])
        sample = [float(sample[i]) for i in get_feature_ordering()]
        scores = get_v_model("vmodel-testing").call(tf.convert_to_tensor([sample]))
        data[player_index] = float(scores.numpy()[0][0])

    return data


@app.route("/games/<string:game_id>", methods=["GET"])
def get_game_endpoint(game_id):
    game = get_last_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    return json.dumps(game, cls=GameEncoder)


@app.route("/games", methods=["POST"])
def create_game():
    game = Game(
        players=[
            # VRLPlayer(Color.RED, "FOO", "models/vp-big-256-64"),
            # TensorRLPlayer(Color.BLUE, "BAR", "tensor-model-normalized"),
            # MCTSPlayer(Color.RED, "FOO", 25),
            RandomPlayer(Color.RED, "FOO"),
            RandomPlayer(Color.BLUE, "BAR"),
            RandomPlayer(Color.WHITE, "BAZ"),
            RandomPlayer(Color.ORANGE, "QUX"),
        ]
    )
    save_game_state(game)
    return jsonify({"game_id": game.id})
