import json
import logging
import traceback
import os
import sys

from flask import Response, Blueprint, jsonify, abort, request
import torch

from catanatron.web.models import upsert_game_state, get_game_state, list_replay_catalog
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.player import Color, RandomPlayer
from catanatron.game import Game
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.web.mcts_analysis import GameAnalyzer
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.gym.envs.action_translator import batch_catanatron_to_capstone
from catanatron.gym.envs.catanatron_env import to_action_space as to_catanatron_action_space

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CAPSTONE_AGENT_PATH = os.path.join(ROOT_DIR, "capstone_agent")
if CAPSTONE_AGENT_PATH not in sys.path:
    sys.path.append(CAPSTONE_AGENT_PATH)

from MainPlayAgent import MainPlayAgent  # noqa: E402
from action_map import describe_action, describe_action_detailed  # noqa: E402
from CONSTANTS import FEATURE_SPACE_SIZE, MAIN_PLAY_AGENT_HIDDEN_SIZE  # noqa: E402

bp = Blueprint("api", __name__, url_prefix="/api")
_MAIN_AGENT_CACHE: dict[str, MainPlayAgent] = {}


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
    if not request.is_json or request.json is None or "players" not in request.json:
        abort(400, description="Missing or invalid JSON body: 'players' key required")
    player_keys = request.json["players"]
    players = list(map(player_factory, zip(player_keys, Color)))

    game = Game(players=players)
    upsert_game_state(game)
    return jsonify({"game_id": game.id})


@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    parsed_state_index = _parse_state_index(state_index)
    game = get_game_state(game_id, parsed_state_index)
    if game is None:
        abort(404, description="Resource not found")

    payload = json.dumps(game, cls=GameEncoder)
    return Response(
        response=payload,
        status=200,
        mimetype="application/json",
    )


@bp.route("/games/<string:game_id>/actions", methods=["POST"])
def post_action_endpoint(game_id):
    game = get_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_color() is not None:
        return Response(
            response=json.dumps(game, cls=GameEncoder),
            status=200,
            mimetype="application/json",
        )

    # TODO: remove `or body_is_empty` when fully implement actions in FE
    body_is_empty = (not request.data) or request.json is None or request.json == {}
    if game.state.current_player().is_bot or body_is_empty:
        game.play_tick()
        upsert_game_state(game)
    else:
        action = action_from_json(request.json)
        game.execute(action)
        upsert_game_state(game)

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/replays", methods=["GET"])
def get_replay_catalog_endpoint():
    limit = request.args.get("limit", default=200, type=int)
    limit = max(1, min(limit, 2000))
    return jsonify({"replays": list_replay_catalog(limit=limit)})


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


@bp.route(
    "/games/<string:game_id>/states/<string:state_index>/mcts-analysis", methods=["GET"]
)
def mcts_analysis_endpoint(game_id, state_index):
    """Get MCTS analysis for specific game state."""
    logging.info(f"MCTS analysis request for game {game_id} at state {state_index}")

    # Convert 'latest' to None for consistency with get_game_state
    parsed_state_index = _parse_state_index(state_index)
    try:
        game = get_game_state(game_id, parsed_state_index)
        if game is None:
            logging.error(
                f"Game/state not found: {game_id}/{state_index}"
            )  # Use original state_index for logging
            abort(404, description="Game state not found")

        analyzer = GameAnalyzer(num_simulations=100)
        probabilities = analyzer.analyze_win_probabilities(game)

        logging.info(f"Analysis successful. Probabilities: {probabilities}")
        return Response(
            response=json.dumps(
                {
                    "success": True,
                    "probabilities": probabilities,
                    "state_index": (
                        parsed_state_index
                        if parsed_state_index is not None
                        else len(game.state.action_records)
                    ),
                }
            ),
            status=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.error(f"Error in MCTS analysis endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return Response(
            response=json.dumps(
                {"success": False, "error": str(e), "trace": traceback.format_exc()}
            ),
            status=500,
            mimetype="application/json",
        )


def _parse_state_index(state_index_str: str):
    """Helper function to parse and validate state_index."""
    if state_index_str == "latest":
        return None
    try:
        return int(state_index_str)
    except ValueError:
        abort(
            400,
            description="Invalid state_index format. state_index must be an integer or 'latest'.",
        )


def _resolve_self_and_opp_colors(game: Game) -> tuple[Color, Color]:
    # Capstone 1v1 convention: our policy is BLUE.
    colors = list(game.state.colors)
    self_color = Color.BLUE if Color.BLUE in colors else colors[0]
    opp_color = colors[0] if colors[0] != self_color else colors[1]
    return self_color, opp_color


def _load_main_agent(model_path: str) -> MainPlayAgent:
    abs_path = os.path.abspath(model_path)
    if abs_path not in _MAIN_AGENT_CACHE:
        agent = MainPlayAgent(
            obs_size=FEATURE_SPACE_SIZE, hidden_size=MAIN_PLAY_AGENT_HIDDEN_SIZE
        )
        agent.load(abs_path)
        _MAIN_AGENT_CACHE[abs_path] = agent
    return _MAIN_AGENT_CACHE[abs_path]


def _action_index_for_record(game, action_record) -> int | None:
    try:
        action = action_record.action
        catan_idx = to_catanatron_action_space(action)
        cap_idxs = batch_catanatron_to_capstone([catan_idx])
        return int(cap_idxs[0]) if cap_idxs else None
    except Exception:
        return None


@bp.route(
    "/games/<string:game_id>/states/<string:state_index>/policy-analysis", methods=["GET"]
)
def policy_analysis_endpoint(game_id, state_index):
    parsed_state_index = _parse_state_index(state_index)
    game = get_game_state(game_id, parsed_state_index)
    if game is None:
        abort(404, description="Game state not found")

    top_n = request.args.get("top_n", default=5, type=int)
    top_n = max(1, min(top_n, 30))
    default_model = os.path.join(ROOT_DIR, "capstone_agent", "models", "challenger_main_play.pt")
    model_path = request.args.get("model_path", default=default_model, type=str)
    if not model_path or not os.path.isfile(model_path):
        abort(400, description=f"model_path does not exist: {model_path}")

    self_color, opp_color = _resolve_self_and_opp_colors(game)
    obs = get_capstone_observation(game, self_color=self_color, opp_color=opp_color)
    valid_catan = [to_catanatron_action_space(a) for a in game.playable_actions]
    valid_capstone = batch_catanatron_to_capstone(valid_catan)
    mask = [0.0] * 245
    for idx in valid_capstone:
        mask[int(idx)] = 1.0

    agent = _load_main_agent(model_path)
    device = agent.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs_t, value_t = agent.model(obs_t, mask_t)
    probs = probs_t.squeeze(0).detach().cpu().tolist()
    state_value = float(value_t.item())

    ranked = sorted(valid_capstone, key=lambda i: probs[int(i)], reverse=True)
    top_idxs = ranked[:top_n]

    top_actions = []
    for idx in top_idxs:
        idx_i = int(idx)
        next_value = None
        try:
            action = game.playable_actions[valid_capstone.index(idx_i)]
            game_next = game.copy()
            game_next.execute(action)
            obs_next = get_capstone_observation(
                game_next, self_color=self_color, opp_color=opp_color
            )
            valid_next = [
                to_catanatron_action_space(a) for a in game_next.playable_actions
            ]
            valid_next_cap = batch_catanatron_to_capstone(valid_next)
            mask_next = [0.0] * 245
            for j in valid_next_cap:
                mask_next[int(j)] = 1.0
            obs_next_t = torch.as_tensor(
                obs_next, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask_next_t = torch.as_tensor(
                mask_next, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                _, value_next_t = agent.model(obs_next_t, mask_next_t)
            next_value = float(value_next_t.item())
        except Exception:
            next_value = None

        top_actions.append(
            {
                "action_index": idx_i,
                "probability": float(probs[idx_i]),
                "description": describe_action(idx_i),
                "description_detailed": describe_action_detailed(idx_i),
                "next_state_value_estimate": next_value,
            }
        )

    chosen_action = None
    if game.state.action_records:
        chosen_idx = _action_index_for_record(game, game.state.action_records[-1])
        if chosen_idx is not None:
            chosen_action = {
                "action_index": chosen_idx,
                "description": describe_action(chosen_idx),
                "description_detailed": describe_action_detailed(chosen_idx),
            }

    return jsonify(
        {
            "success": True,
            "state_index": int(parsed_state_index)
            if parsed_state_index is not None
            else len(game.state.action_records),
            "model_path": os.path.abspath(model_path),
            "top_n": top_n,
            "state_value_estimate": state_value,
            "top_actions": top_actions,
            "chosen_action": chosen_action,
        }
    )


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
