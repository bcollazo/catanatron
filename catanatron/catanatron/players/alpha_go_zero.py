from catanatron.game import Game
from catanatron.models.player import Player


class AlphaGoZeroPlayer(Player):
    """
    Combined policy and value network for Catan, inspired by AlphaGo Zero.


    Uses PUCT formula for
    """

    def __init__(self, color):
        super().__init__(color)

        # create NN

    def decide(self, game: Game, playable_actions):
        pass
