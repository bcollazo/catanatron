"""Player wrapper: uniform-random during initial build, then delegate to another player."""

import random

from catanatron.models.player import Player


class InitialBuildRandomPlayer(Player):
    """Random choice among legal actions only while `game.state.is_initial_build_phase` is true."""

    def __init__(self, color, inner: Player):
        super().__init__(color, is_bot=True)
        self.inner = inner

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if not actions:
            raise ValueError("InitialBuildRandomPlayer: no playable actions")
        if game.state.is_initial_build_phase:
            return random.choice(actions)
        return self.inner.decide(game, playable_actions)

    def reset_state(self):
        if hasattr(self.inner, "reset_state"):
            self.inner.reset_state()

    def __repr__(self):
        return f"InitialBuildRandom({self.inner!r})"
