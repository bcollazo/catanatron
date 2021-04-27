import json

from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from dotenv import load_dotenv

from catanatron_server.database import save_game_state, get_last_game_state
from catanatron.game import Game
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.player import RandomPlayer, Color

from experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    ValueFunctionPlayer,
)

load_dotenv()  # useful if running server outside docker
app = Flask(__name__)
CORS(app)


BOT_COLOR = Color.RED


def advance_until_color(game, color, callbacks):
    while game.winning_player() is None and game.state.current_player().color != color:
        game.play_tick(callbacks)


@app.route("/games/<string:game_id>/actions", methods=["POST"])
def tick_game(game_id):
    game = get_last_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_player() is not None:
        return json.dumps(game, cls=GameEncoder)

    bots_turn = game.state.current_player().color == BOT_COLOR
    if bots_turn or request.json is None:  # TODO: Remove this or
        game.play_tick([lambda g: save_game_state(g)])
    else:
        action = action_from_json(request.json)
        game.execute(action, action_callbacks=[lambda g: save_game_state(g)])

    # advance_until_color(game, Color.BLUE, [lambda g: save_game_state(g)])
    return json.dumps(game, cls=GameEncoder)


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
            AlphaBetaPlayer(BOT_COLOR, "FOO"),
            ValueFunctionPlayer(Color.BLUE, "BAR"),
        ]
    )
    save_game_state(game)
    # advance_until_color(game, Color.BLUE, [lambda g: save_game_state(g)])
    return jsonify({"game_id": game.id})


# ===== Debugging Routes
# @app.route(
#     "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
# )
# def get_game_feature_vector(game_id, player_index):
#     game = get_last_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     return create_sample(game, game.state.players[player_index].color)


# @app.route("/games/<string:game_id>/value-function", methods=["GET"])
# def get_game_value_function(game_id):
#     game = get_last_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     # model = tf.keras.models.load_model("experimental/models/mcts-rep-a")
#     model2 = tf.keras.models.load_model("experimental/models/mcts-rep-b")
#     feature_ordering = get_feature_ordering()
#     indices = [feature_ordering.index(f) for f in NUMERIC_FEATURES]
#     data = {}
#     for player in game.state.players:
#         sample = create_sample_vector(game, player.color)
#         # scores = model.call(tf.convert_to_tensor([sample]))

#         inputs1 = [create_board_tensor(game, player.color)]
#         inputs2 = [[float(sample[i]) for i in indices]]
#         scores2 = model2.call(
#             [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
#         )
#         data[player.color.value] = float(scores2.numpy()[0][0])

#     return data
