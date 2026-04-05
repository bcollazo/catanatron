import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .CONSTANTS import FEATURE_SPACE_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE
    from .PPOHyperparams import PPOHyperparams
    from .PlacementModel import PlacementModel
    from .RolloutBuffer import RolloutBuffer
    from .device import get_device
    from .placement_action_space import (
        PlacementPrompt,
        capstone_action_to_local,
        capstone_mask_to_local_mask,
        infer_placement_prompt,
        local_action_size,
        local_action_to_capstone,
    )
    from .placement_features import (
        COMPACT_PLACEMENT_FEATURE_SIZE,
        project_capstone_batch_to_compact_placement,
        project_capstone_to_compact_placement,
    )
except ImportError:  # pragma: no cover - supports script-style imports
    from CONSTANTS import FEATURE_SPACE_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE
    from PPOHyperparams import PPOHyperparams
    from PlacementModel import PlacementModel
    from RolloutBuffer import RolloutBuffer
    from device import get_device
    from placement_action_space import (
        PlacementPrompt,
        capstone_action_to_local,
        capstone_mask_to_local_mask,
        infer_placement_prompt,
        local_action_size,
        local_action_to_capstone,
    )
    from placement_features import (
        COMPACT_PLACEMENT_FEATURE_SIZE,
        project_capstone_batch_to_compact_placement,
        project_capstone_to_compact_placement,
    )

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

    def select_action(self, state, mask, **_kwargs):
        valid = np.where(np.asarray(mask) > 0.5)[0]
        if len(valid) == 0:
            raise ValueError("RandomPlacementAgent received a mask with no valid actions")
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


class PlacementRolloutBuffer(RolloutBuffer):
    """Placement replay buffer storing prompt-specific local actions."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def store(self, state, mask, action, log_prob, reward, value, done, prompt):
        super().store(state, mask, action, log_prob, reward, value, done)
        self.prompts.append(int(prompt))

    def clear(self):
        """Clear base rollout fields and prompt labels explicitly."""
        super().clear()
        self.prompts = []


@register_strategy("model")
class PlacementAgent:
    """Agent for initial settlement + road placement.

    Shares the same select_action / store / train interface as
    MainPlayAgent so the CapstoneAgent can delegate transparently.

    Public callers still pass the full Capstone observation vector and the
    245-d action mask.  Internally, the agent projects those down to a compact
    opening-only feature space and a prompt-specific local action space.
    """

    MAX_LOCAL_ACTIONS = PlacementModel.EDGE_ACTION_SIZE

    def __init__(
        self,
        obs_size=FEATURE_SPACE_SIZE,
        hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
    ):
        del obs_size  # Kept for backwards compatibility with existing callers.

        self.device = get_device()
        self.hyperparams = PPOHyperparams()
        self.hyperparams.batch_size = 16

        self.model = PlacementModel(
            obs_size=COMPACT_PLACEMENT_FEATURE_SIZE,
            hidden_size=hidden_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams.lr
        )
        self.buffer = PlacementRolloutBuffer()

    def _compact_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"Expected a 1-d state vector, got shape {state.shape}")
        if state.shape[0] == COMPACT_PLACEMENT_FEATURE_SIZE:
            return state.copy()
        if state.shape[0] == FEATURE_SPACE_SIZE:
            return project_capstone_to_compact_placement(state)
        raise ValueError(
            "Unsupported placement state width "
            f"{state.shape[0]} (expected {FEATURE_SPACE_SIZE} or "
            f"{COMPACT_PLACEMENT_FEATURE_SIZE})"
        )

    def _prepare_public_inputs(self, state, mask):
        full_mask = np.asarray(mask, dtype=np.float32)
        if full_mask.ndim != 1:
            raise ValueError(f"Expected a 1-d action mask, got shape {full_mask.shape}")

        prompt = infer_placement_prompt(full_mask)
        compact_state = self._compact_state(state)
        local_mask = capstone_mask_to_local_mask(full_mask, prompt)
        padded_mask = np.zeros(self.MAX_LOCAL_ACTIONS, dtype=np.float32)
        size = local_action_size(prompt)
        padded_mask[:size] = local_mask
        return compact_state, prompt, padded_mask

    def _prepare_supervised_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"Expected a 2-d observation batch, got shape {obs.shape}")

        if obs.shape[1] == FEATURE_SPACE_SIZE:
            return project_capstone_batch_to_compact_placement(obs)
        if obs.shape[1] == COMPACT_PLACEMENT_FEATURE_SIZE:
            return obs.copy()
        raise ValueError(
            "Unsupported placement observation width "
            f"{obs.shape[1]} (expected {FEATURE_SPACE_SIZE} or "
            f"{COMPACT_PLACEMENT_FEATURE_SIZE})"
        )

    def _prepare_supervised_actions_and_masks(self, masks, actions):
        masks = np.asarray(masks, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        if masks.ndim != 2:
            raise ValueError(f"Expected a 2-d mask batch, got shape {masks.shape}")

        prompts = np.empty(len(masks), dtype=np.int64)
        local_masks = np.zeros((len(masks), self.MAX_LOCAL_ACTIONS), dtype=np.float32)
        local_actions = np.empty(len(masks), dtype=np.int64)

        for idx, full_mask in enumerate(masks):
            prompt = infer_placement_prompt(full_mask)
            prompts[idx] = int(prompt)
            prompt_mask = capstone_mask_to_local_mask(full_mask, prompt)
            prompt_size = local_action_size(prompt)
            local_masks[idx, :prompt_size] = prompt_mask
            local_actions[idx] = capstone_action_to_local(prompt, actions[idx])

        return prompts, local_masks, local_actions

    def _dist_for_single_prompt(
        self,
        settlement_logits,
        road_logits,
        prompt_id,
        masks,
    ):
        """Create a categorical distribution for one prompt-specific batch."""

        prompt = PlacementPrompt(int(prompt_id))
        if prompt == PlacementPrompt.SETTLEMENT:
            logits = settlement_logits
            logits_mask = masks[:, :local_action_size(prompt)]
        else:
            logits = road_logits
            logits_mask = masks[:, :local_action_size(prompt)]

        masked_logits = logits.masked_fill(logits_mask <= 0, -1e9)
        return Categorical(logits=masked_logits)

    def _evaluate_actions(self, states, prompts, masks, actions):
        settlement_logits, road_logits, values = self.model(states)
        log_probs = torch.empty(len(states), device=self.device)
        entropies = torch.empty(len(states), device=self.device)

        for prompt in (PlacementPrompt.SETTLEMENT, PlacementPrompt.ROAD):
            idx = prompts == int(prompt)
            if not torch.any(idx):
                continue

            if prompt == PlacementPrompt.SETTLEMENT:
                logits = settlement_logits[idx]
            else:
                logits = road_logits[idx]

            prompt_masks = masks[idx, :local_action_size(prompt)]
            masked_logits = logits.masked_fill(prompt_masks <= 0, -1e9)
            dist = Categorical(logits=masked_logits)

            log_probs[idx] = dist.log_prob(actions[idx])
            entropies[idx] = dist.entropy()

        return log_probs, entropies, values

    def select_action(self, state, mask, **_kwargs):
        """Sample an action from the policy and return (action, log_prob, value)."""
        compact_state, prompt, local_mask = self._prepare_public_inputs(state, mask)

        state_tensor = torch.from_numpy(compact_state).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(local_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            settlement_logits, road_logits, value = self.model(state_tensor)
            dist = self._dist_for_single_prompt(
                settlement_logits,
                road_logits,
                int(prompt),
                mask_tensor,
            )

        action = dist.sample()
        log_prob = dist.log_prob(action)
        capstone_action = local_action_to_capstone(prompt, action.item())

        return (capstone_action, log_prob.item(), value.item())

    def store(self, state, mask, action, log_prob, reward, value, done):
        compact_state, prompt, local_mask = self._prepare_public_inputs(state, mask)
        local_action = capstone_action_to_local(prompt, action)
        self.buffer.store(
            compact_state,
            local_mask,
            local_action,
            log_prob,
            reward,
            value,
            done,
            int(prompt),
        )

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
        prompts_np = np.asarray(self.buffer.prompts, dtype=np.int64)
        old_log_probs_np = np.asarray(self.buffer.log_probs, dtype=np.float32)

        states = torch.from_numpy(states_np).to(self.device)
        masks = torch.from_numpy(masks_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        prompts = torch.from_numpy(prompts_np).to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(self.device)

        for epoch in range(self.hyperparams.epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.hyperparams.batch_size):
                batch_idx = indices[start : start + self.hyperparams.batch_size]

                b_states = states[batch_idx]
                b_masks = masks[batch_idx]
                b_actions = actions[batch_idx]
                b_prompts = prompts[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                new_log_probs, entropies, values = self._evaluate_actions(
                    b_states,
                    b_prompts,
                    b_masks,
                    b_actions,
                )
                entropy = entropies.mean()

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
            obs:     Either full Capstone observations ``(N, 1259)`` or
                     compact placement observations
                     ``(N, COMPACT_PLACEMENT_FEATURE_SIZE)``.
            masks:   Full Capstone action masks ``(N, 245)``.
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

        compact_obs = self._prepare_supervised_obs(obs)
        prompt_ids, local_masks, local_actions = self._prepare_supervised_actions_and_masks(
            masks, actions
        )

        obs_t = torch.as_tensor(compact_obs, dtype=torch.float32).to(self.device)
        mask_t = torch.as_tensor(local_masks, dtype=torch.float32).to(self.device)
        act_t = torch.as_tensor(local_actions, dtype=torch.long).to(self.device)
        prompt_t = torch.as_tensor(prompt_ids, dtype=torch.long).to(self.device)
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
                b_prompt = prompt_t[idx]
                b_w = weights[idx]

                log_probs, _, _ = self._evaluate_actions(
                    b_obs,
                    b_prompt,
                    b_mask,
                    b_act,
                )
                loss = (-log_probs * b_w).mean()

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


HeavyPlacementAgent = PlacementAgent
