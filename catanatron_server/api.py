import json

from flask import Response, Blueprint, jsonify, abort, request

from catanatron_server.models import create_game_state, get_game_state
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.player import Color
from catanatron.game import Game
from experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    ValueFunctionPlayer,
)

BOT_COLOR = Color.RED

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/games", methods=("POST",))
def post_game_endpoint():
    game = Game(
        players=[
            AlphaBetaPlayer(BOT_COLOR, "FOO", 2, True),
            ValueFunctionPlayer(Color.BLUE, "BAR"),
        ]
    )
    create_game_state(game)
    return jsonify({"game_id": game.id})


@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    state_index = None if state_index == "latest" else int(state_index)
    game = get_game_state(game_id, state_index)
    if game is None:
        abort(404, description="Resource not found")

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/games/<string:game_id>/actions", methods=["POST"])
def post_action_endpoint(game_id):
    game = get_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_player() is not None:
        return Response(
            response=json.dumps(game, cls=GameEncoder),
            status=200,
            mimetype="application/json",
        )

    bots_turn = game.state.current_player().color == BOT_COLOR
    if bots_turn or request.json is None:  # TODO: Remove this or
        game.play_tick([lambda g: create_game_state(g)])
    else:
        action = action_from_json(request.json)
        game.execute(action, action_callbacks=[lambda g: create_game_state(g)])

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


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
