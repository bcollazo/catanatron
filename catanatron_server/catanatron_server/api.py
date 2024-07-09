import json

from flask import Response, Blueprint, jsonify, abort, request

from catanatron_server.models import get_game_metadata, get_games_info, serialize_game_state, load_game_state
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.player import Color, RandomPlayer
from catanatron.game import Game
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_gym.features import (
    create_sample,
    # create_sample_vector,
    # get_feature_ordering,
)

bp = Blueprint("api", __name__, url_prefix="/api")


def player_factory(player_key):
    if player_key[0] == "CATANATRON":
        return AlphaBetaPlayer(player_key[1], 2, True)
    elif player_key[0] == "RANDOM":
        return RandomPlayer(player_key[1])
    elif player_key[0] == "HUMAN":
        return ValueFunctionPlayer(player_key[1], is_bot=False)
    else:
        raise ValueError("Invalid player key")


@bp.route("/games", methods=("POST",))
def post_game_endpoint():
    player_keys = request.json["players"]
    players = list(map(player_factory, zip(player_keys, Color)))

    game = Game(players=players)
    serialize_game_state(game, game.state_index)
    
    return jsonify({
        "game_id": game.id,
        "state_index": game.state_index
        })

# get game state by uuid and state_index
@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    state_index = None if state_index == "latest" else int(state_index)
    game = load_game_state(game_id, state_index)
    if game is None:
        abort(404, description="Resource not found")

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/games/<string:game_id>/states/<string:state_index>/actions", methods=["POST"])
def post_action_endpoint(game_id, state_index):
    state_index = None if state_index == "latest" else int(state_index)

    game = load_game_state(game_id, state_index)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_color() is not None:
        return Response(
            response=json.dumps(game, cls=GameEncoder),
            status=200,
            mimetype="application/json",
        )

    # TODO: remove `or body_is_empty` when fully implement actions in FE
    body_is_empty = (not request.data) or request.json is None
    if game.state.current_player().is_bot or body_is_empty:
        game.play_tick()
        serialize_game_state(game, game.state_index)
    else:
        action = action_from_json(request.json)
        game.execute(action)
        serialize_game_state(game, game.state_index)

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/stress-test", methods=["GET"])
def stress_test_endpoint():
    players = [
        AlphaBetaPlayer(Color.RED, 2, True),
        AlphaBetaPlayer(Color.BLUE, 2, True),
        AlphaBetaPlayer(Color.ORANGE, 2, True),
        AlphaBetaPlayer(Color.WHITE, 2, True),
    ]
    game = Game(players=players)
    game.play_tick()
    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


# ===== Debugging Routes
@bp.route(
    "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
)
def get_game_feature_vector(game_id, player_index):
    # game = get_game(game_id)
    game = load_game_state(game_id, None)
    if game is None:
        abort(404, description="Resource not found")

    return create_sample(game, game.state.colors[player_index])


# get game info
@bp.route(
    "/games/<string:game_id>/info", methods=["GET"]
)
def get_game_info(game_id):
    game_metadata = get_game_metadata(game_id)

    if game_metadata is None:
        abort(404, description="Metadata not found")

    return jsonify({
        "game_states_count": game_metadata["game_states_count"],
        "winner": game_metadata["winner"],
        "players": game_metadata["players"]
    })

# get general info
@bp.route(
    "/info", methods=["GET"]
)
def get_info():
    game_count, games_uuid = get_games_info()

    # games_uuid = [str(game_id[0]) for game_id in games_uuid] # convert to string

    return jsonify({
        # "game_states_count": game_states_count, 
        "games_count": game_count, 
        "games_uuid": games_uuid
        })


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
