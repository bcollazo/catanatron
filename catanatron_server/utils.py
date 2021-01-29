from catanatron_server.database import save_game_state


def open_game(game):
    save_game_state(game)
    print("http://localhost:3000/games/" + game.id)
