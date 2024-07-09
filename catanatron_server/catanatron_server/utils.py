import webbrowser

def open_link(game):
    """opens game in browser"""
    link = f"http://localhost:3000/games/{game.id}/states/latest"
    webbrowser.open(link)
