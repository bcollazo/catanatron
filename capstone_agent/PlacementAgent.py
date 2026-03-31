from PlacementModel import PlacementModel
from RolloutBuffer import RolloutBuffer
from PPOHyperparams import PPOHyperparams
from device import get_device

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from CONSTANTS import FEATURE_SPACE_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE

# ── registry of available strategies ─────────────────────────────

PLACEMENT_STRATEGIES = {}


def register_strategy(name):
    """Decorator that adds a class to the PLACEMENT_STRATEGIES dict."""
    def wrapper(cls):
        PLACEMENT_STRATEGIES[name] = cls
        return cls
    return wrapper


def make_placement_agent(strategy: str = "model", **kwargs):
    """Construct a placement agent by strategy name.

    Args:
        strategy: One of the keys in PLACEMENT_STRATEGIES
                  (currently ``"model"`` or ``"random"``).
        **kwargs: Forwarded to the chosen class constructor.

    Returns:
        An agent that implements select_action / store / train / load / save.
    """
    if strategy not in PLACEMENT_STRATEGIES:
        available = ", ".join(sorted(PLACEMENT_STRATEGIES))
        raise ValueError(
            f"Unknown placement strategy {strategy!r}. Choose from: {available}"
        )
    return PLACEMENT_STRATEGIES[strategy](**kwargs)


# ── random baseline ──────────────────────────────────────────────

@register_strategy("random")
class RandomPlacementAgent:
    """Picks uniformly at random from valid actions.  No model, no training."""

    def __init__(self, **_kwargs):
        pass

    def select_action(self, state, mask):
        valid = np.where(np.asarray(mask) > 0.5)[0]
        action = int(np.random.choice(valid))
        n_valid = len(valid)
        log_prob = -math.log(n_valid) if n_valid > 0 else 0.0
        value = 0.0
        return (action, log_prob, value)

    def store(self, state, mask, action, log_prob, reward, value, done):
        pass

    def train(self, last_value):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


# ── learned placement agent ─────────────────────────────────────

@register_strategy("model")
class PlacementAgent:
    """Agent for initial settlement + road placement.

    Shares the same select_action / store / train interface as
    MainPlayAgent so the CapstoneAgent can delegate transparently.

    Uses a lightweight PlacementModel (settlement + road heads only,
    2 residual blocks, ~0.8M params vs 6.3M for the full model).
    Supports both online PPO updates and offline supervised training
    via ``supervised_train()``.
    """

    def __init__(self, obs_size=FEATURE_SPACE_SIZE, hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE): # TODO -> does this really need the whole feature space?
        self.device = get_device()
        self.hyperparams = PPOHyperparams()
        self.hyperparams.batch_size = 16

        self.model = PlacementModel(obs_size=obs_size, hidden_size=hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams.lr
        )
        self.buffer = RolloutBuffer()

    def select_action(self, state, mask):
        """Sample an action from the policy and return (action, log_prob, value)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, value = self.model(state_tensor, mask_tensor)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (action.item(), log_prob.item(), value.item())

    def store(self, state, mask, action, log_prob, reward, value, done):
        self.buffer.store(state, mask, action, log_prob, reward, value, done)

    def compute_advantages(self, last_value):
        advantages = []
        gae = 0

        values = self.buffer.values + [last_value]
        rewards = self.buffer.rewards
        dones = self.buffer.dones

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.hyperparams.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            gae = (
                delta
                + self.hyperparams.gamma
                * self.hyperparams.gae_lambda
                * (1 - dones[t])
                * gae
            )
            advantages.append(gae)

        advantages.reverse()

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.buffer.values).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train(self, last_value):
        """PPO update over collected placement transitions."""
        if len(self.buffer.rewards) == 0:
            return

        advantages, returns = self.compute_advantages(last_value)

        states_np = np.asarray(self.buffer.states, dtype=np.float32)
        masks_np = np.asarray(self.buffer.masks, dtype=np.float32)
        actions_np = np.asarray(self.buffer.actions, dtype=np.int64)
        old_log_probs_np = np.asarray(self.buffer.log_probs, dtype=np.float32)

        states = torch.from_numpy(states_np).to(self.device)
        masks = torch.from_numpy(masks_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(self.device)

        for epoch in range(self.hyperparams.epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.hyperparams.batch_size):
                batch_idx = indices[start : start + self.hyperparams.batch_size]

                b_states = states[batch_idx]
                b_masks = masks[batch_idx]
                b_actions = actions[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                probs, values = self.model(b_states, b_masks)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.hyperparams.clip_eps,
                        1 + self.hyperparams.clip_eps,
                    )
                    * b_advantages
                )
                actor_loss = -torch.where(
                    b_advantages >= 0,
                    torch.min(surr1, surr2),
                    torch.max(surr1, surr2),
                ).mean()

                critic_loss = nn.MSELoss()(values.view(-1), b_returns.view(-1))

                loss = (
                    actor_loss
                    + self.hyperparams.value_coef * critic_loss
                    - self.hyperparams.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.hyperparams.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.clear()

    # ── supervised learning interface ────────────────────────────

    def supervised_train(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        won: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        win_weight: float = 1.0,
        loss_weight: float = 0.1,
    ) -> list:
        """Train on a labelled dataset of placement decisions.

        Args:
            obs:     (N, 1259)  board observations at placement time.
            masks:   (N, 245)   action masks at placement time.
            actions: (N,)       action indices that were chosen.
            won:     (N,)       1.0 if the acting player won, else 0.0.
            epochs:  Number of full passes over the dataset.
            batch_size: Mini-batch size.
            lr:      Learning rate (resets the optimizer).
            win_weight:  Sample weight for winning games.
            loss_weight: Sample weight for losing games.

        Returns:
            List of per-epoch mean losses.
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.float32).to(self.device)
        act_t = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        weights = torch.where(
            torch.as_tensor(won, dtype=torch.float32) > 0.5,
            win_weight,
            loss_weight,
        ).to(self.device)

        n = len(obs_t)
        epoch_losses = []

        for epoch in range(epochs):
            perm = torch.randperm(n)
            running_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                b_obs = obs_t[idx]
                b_mask = mask_t[idx]
                b_act = act_t[idx]
                b_w = weights[idx]

                probs, _ = self.model(b_obs, b_mask)
                log_probs = torch.log(probs + 1e-8)
                nll = nn.functional.nll_loss(log_probs, b_act, reduction="none")
                loss = (nll * b_w).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg = running_loss / max(n_batches, 1)
            epoch_losses.append(avg)

        self.model.eval()
        return epoch_losses

    # ── persistence ──────────────────────────────────────────────

    def load(self, path):
        self.model.load_state_dict(
            torch.load(path, weights_only=True, map_location=self.device)
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)
