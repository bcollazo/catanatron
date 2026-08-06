from catanatron.models.observation import Observation
from catanatron.models.player import Player


class PerspectivePlayer(Player):
    """Player that only ever perceives an Observation, never a Game.

    Subclass and implement decide_observation. The fairness contract is
    structural: a PerspectivePlayer never receives a Game.
    """

    def decide(self, game, playable_actions):
        return self.decide_observation(
            Observation(game, self.color), playable_actions
        )

    def decide_observation(self, observation, playable_actions):
        raise NotImplementedError
