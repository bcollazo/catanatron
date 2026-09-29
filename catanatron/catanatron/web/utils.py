import webbrowser

from catanatron.web.models import database_session, upsert_game_state


def ensure_link(game, get_replay_link: bool = False):
    """Upserts game to database per DATABASE_URL

    Returns:
        str: URL for inspecting state, per convention
    """
    with database_session() as session:
        stored = upsert_game_state(game, session)
        if get_replay_link:
            url = f"http://localhost:3000/replays/{stored.uuid}"
        else:
            url = (
                f"http://localhost:3000/games/{stored.uuid}/states/{stored.head_index}"
            )

    return url


def open_link(game):
    """Upserts game to database and opens game in browser"""
    link = ensure_link(game)
    webbrowser.open(link)
