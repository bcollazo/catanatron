from catanatron_server.database import save_game_state
from catanatron_server.models import database_session, create_game_state


def open_game(game):
    save_game_state(game)
    print("http://localhost:3000/games/" + game.id)


def ensure_link(game):
    with database_session() as session:
        game_state = create_game_state(game, session)
        url = f"http://localhost:3000/games/{game_state.uuid}/states/{game_state.state_index}"
    return url
