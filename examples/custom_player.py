from typing import Iterable

from catanatron import Game, Action, Player
from catanatron.cli import register_cli_player


class FooPlayer(Player):
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
        return playable_actions[0]  # type: ignore
        # ===== END YOUR CODE =====


register_cli_player("FOO", FooPlayer)
