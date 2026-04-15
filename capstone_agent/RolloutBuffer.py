class RolloutBuffer:
    """Stores one episode/rollout worth of data, then gets cleared"""
    def __init__(self):
        self.states       = []
        self.masks        = []
        self.actions      = []
        self.log_probs    = []   # log prob of action taken (for ratio calc)
        self.rewards      = []
        self.values       = []   # critic estimates
        self.dones        = []

    def store(self, state, mask, action, log_prob, reward, value, done):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def extend_from_lists(
        self,
        states,
        masks,
        actions,
        log_probs,
        rewards,
        values,
        dones,
    ) -> None:
        """Append transitions from plain lists (pickled from worker processes)."""
        self.states.extend(states)
        self.masks.extend(masks)
        self.actions.extend(actions)
        self.log_probs.extend(log_probs)
        self.rewards.extend(rewards)
        self.values.extend(values)
        self.dones.extend(dones)