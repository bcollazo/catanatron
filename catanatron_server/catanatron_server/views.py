import json
from flask import Response, abort, jsonify
from .utils import (
    get_game_state_from_db,
    save_game_state_to_db,
    player_factory,
)
from catanatron_server.models import GameEncoder
from catanatron.models.player import Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.json import action_from_json
from catanatron.game import Game


def get_game_view(game_id, state_index):
    state_index = None if state_index == "latest" else int(state_index)
    game = get_game_state_from_db(game_id, state_index)
    if game is None:
        abort(404, description="Resource not found")

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


def post_game_view(player_keys):
    players = list(map(player_factory, zip(player_keys, Color)))

    game = Game(players=players)
    save_game_state_to_db(game, game.state_index)

    return jsonify({"game_id": game.id, "state_index": game.state_index})


def post_action_view(game_id, request):
    game = get_game_state_from_db(game_id)
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
        save_game_state_to_db(game)
    else:
        action = action_from_json(request.json)
        game.execute(action)
        save_game_state_to_db(game)

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


def stress_test_view():
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
