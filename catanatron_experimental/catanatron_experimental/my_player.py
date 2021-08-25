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
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        return playable_actions[0]
        # ===== END YOUR CODE =====
