from flask import Blueprint, request

from catanatron_server.views import (
    get_game_view,
    post_game_view,
    stress_test_view,
    post_action_view,
)


bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/games", methods=("POST",))
def post_game_endpoint():
    player_keys = request.json["players"]

    return post_game_view(player_keys)


@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    return get_game_view(game_id, state_index)


@bp.route("/games/<string:game_id>/actions", methods=["POST"])
def post_action_endpoint(game_id):

    return post_action_view(game_id, request)


@bp.route("/stress-test", methods=["GET"])
def stress_test_endpoint():
    return stress_test_view()


# ===== Debugging Routes
# @app.route(
#     "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
# )
# def get_game_feature_vector(game_id, player_index):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     return create_sample(game, game.state.colors[player_index])


# @app.route("/games/<string:game_id>/value-function", methods=["GET"])
# def get_game_value_function(game_id):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     # model = tf.keras.models.load_model("data/models/mcts-rep-a")
#     model2 = tf.keras.models.load_model("data/models/mcts-rep-b")
#     feature_ordering = get_feature_ordering()
#     indices = [feature_ordering.index(f) for f in NUMERIC_FEATURES]
#     data = {}
#     for color in game.state.colors:
#         sample = create_sample_vector(game, color)
#         # scores = model.call(tf.convert_to_tensor([sample]))

#         inputs1 = [create_board_tensor(game, color)]
#         inputs2 = [[float(sample[i]) for i in indices]]
#         scores2 = model2.call(
#             [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
#         )
#         data[color.value] = float(scores2.numpy()[0][0])

#     return data
