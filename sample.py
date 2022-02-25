from catanatron import Game, RandomPlayer, Color

from catanatron_experimental.my_player import MyPlayer
from catanatron_server.utils import open_link

# Play a simple 4v4 game. Edit MyPlayer with your logic!
players = [
    MyPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())  # returns winning color

# Ensure you have `docker-compose up` running
#   in another terminal tab:
open_link(game)  # opens game result in browser
