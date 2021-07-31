from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron_experimental.my_player import MyPlayer


players = [
    MyPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())  # returns winning color

from catanatron_server.utils import open_link

open_link(game)
