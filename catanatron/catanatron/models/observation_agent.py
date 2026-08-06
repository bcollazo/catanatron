"""
A fair bot that decides from Observations, never from a Game.

An ObservationAgent is deployment-agnostic: the same class is driven by the
local engine (through a PerspectivePlayer adapter) and by a future site-mode
adapter that feeds observations built from external events.
"""


class ObservationAgent:
    """Base class for agents that only ever perceive Observations.

    Subclass and implement decide_observation. The fairness contract is
    structural: an ObservationAgent never receives a Game.
    """

    def __init__(self, color):
        self.color = color

    def decide_observation(self, observation, playable_actions):
        raise NotImplementedError

    def reset_state(self):
        """Hook for resetting memory between games."""
        pass
