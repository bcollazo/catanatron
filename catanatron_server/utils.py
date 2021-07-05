import webbrowser

from catanatron_server.models import database_session, upsert_game_state


def ensure_link(game):
    with database_session() as session:
        game_state = upsert_game_state(game, session)
        url = f"http://localhost:3000/games/{game_state.uuid}/states/{game_state.state_index}"
    return url


def open_link(game):
    link = ensure_link(game)
    webbrowser.open(link)
