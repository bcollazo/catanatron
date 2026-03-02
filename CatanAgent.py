from catanatron import Player
from catanatron.cli import register_cli_player

class RLCatanPlayer(Player):
    def decide(self, game, playable_actions):
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

register_cli_player("Capstone_RL_AGENT", RLCatanPlayer)
