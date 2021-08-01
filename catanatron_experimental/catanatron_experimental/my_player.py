from typing import Iterable

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player


class MyPlayer(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # returns first action, probably can be smarter about this
        # edit this code and implement your bot here!
        return playable_actions[0]
