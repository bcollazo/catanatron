import torch


def _unwrap_env(env):
    """Walk through gymnasium wrappers to reach the core env."""
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


class AgentRouter:
    """Routes decisions to a PlacementAgent during the initial build phase
    and to the main CapstoneAgent for the rest of the game.

    Presents the same interface as CapstoneAgent / PlacementAgent so it can
    be used as a drop-in replacement in simulate_game / training_loop.
    """

    def __init__(self, placement_agent, main_agent, env):
        self.placement_agent = placement_agent
        self.main_agent = main_agent
        self.env = env
        self._last_was_placement = False

    def _is_placement_phase(self) -> bool:
        core_env = _unwrap_env(self.env)
        return core_env.game.state.is_initial_build_phase

    def select_action(self, state, mask):
        self._last_was_placement = self._is_placement_phase()
        if self._last_was_placement:
            return self.placement_agent.select_action(state, mask)
        return self.main_agent.select_action(state, mask)

    def store(self, state, mask, action, log_prob, reward, value, done):
        if self._last_was_placement:
            self.placement_agent.store(state, mask, action, log_prob, reward, value, done)
        else:
            self.main_agent.store(state, mask, action, log_prob, reward, value, done)

    def train(self, last_value):
        """Run PPO updates on both sub-agents (placement only if it has data)."""
        self.main_agent.train(last_value)
        self.placement_agent.train(last_value)

    @property
    def model(self):
        """Convenience accessor used by simulate_and_train to compute last_value.

        Returns the main agent's model since last_value is needed for the
        main-game value estimate (placement is already over by end-of-game).
        """
        return self.main_agent.model

    def load(self, main_path, placement_path=None):
        self.main_agent.load(main_path)
        if placement_path is not None:
            self.placement_agent.load(placement_path)

    def save(self, main_path, placement_path=None):
        self.main_agent.save(main_path)
        if placement_path is not None:
            self.placement_agent.save(placement_path)
