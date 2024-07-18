import pickle
import webbrowser
from catanatron_server.models import db, GameState
from catanatron.models.player import RandomPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

from catanatron_server.models import database_session


# serialize and store game state using db
def save_game_state_to_db(game, session_param=None):
    game_state = GameState.from_game(game)
    
    session = session_param or db.session
    session.add(game_state)
    session.commit()
    
    return game_state


def get_game_state_from_db(game_id, state_index=None):
    if state_index is None:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id)
            .order_by(GameState.state_index.desc())
            .first_or_404()
        )
    else:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id, state_index=state_index)
            .first_or_404()
        )
    db.session.commit()
    game = pickle.loads(result.pickle_data)
    return game


def player_factory(player_key):
    if player_key[0] == "CATANATRON":
        return AlphaBetaPlayer(player_key[1], 2, True)
    elif player_key[0] == "RANDOM":
        return RandomPlayer(player_key[1])
    elif player_key[0] == "HUMAN":
        return ValueFunctionPlayer(player_key[1], is_bot=False)
    else:
        raise ValueError("Invalid player key")


def ensure_link(game):
    """Upserts game to database per DATABASE_URL

    Returns:
        str: URL for inspecting state, per convention
    """
    with database_session() as session:
        game_state = save_game_state_to_db(game, session)
        url = f"http://localhost:3000/games/{game_state.uuid}/states/{game_state.state_index}"
    return url


def open_link(game):
    """Upserts game to database and opens game in browser"""
    link = ensure_link(game)
    webbrowser.open(link)
