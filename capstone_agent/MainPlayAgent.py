from MainPlayModel import MainPlayModel
from RolloutBuffer import RolloutBuffer
from PPOHyperparams import PPOHyperparams
from device import get_device

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from CONSTANTS import FEATURE_SPACE_SIZE, MAIN_PLAY_AGENT_HIDDEN_SIZE
from typing import Dict


class MainPlayAgent:

    def __init__(self, obs_size=FEATURE_SPACE_SIZE, hidden_size=MAIN_PLAY_AGENT_HIDDEN_SIZE):
        self.device = get_device()
        self.hyperparams = PPOHyperparams()
        self.model = MainPlayModel(obs_size=obs_size, hidden_size=hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams.lr)
        self.buffer = RolloutBuffer()


    def select_action(self, state, mask, **_kwargs):
        """
        Given state + mask, sample an action and return 
        everything PPO needs to store.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor  = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, value = self.model(state_tensor, mask_tensor)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # log prob and value are used for training

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )
    
    def store(self, state, mask, action, log_prob, reward, value, done, next_obs=None):
        self.buffer.store(state, mask, action, log_prob, reward, value, done)

    def compute_advantages(self, last_value):
        """
        GAE (Generalized Advantage Estimation) — 
        balances bias vs variance in advantage estimates
        """
        advantages = []
        gae = 0

        values  = self.buffer.values + [last_value]
        rewards = self.buffer.rewards
        dones   = self.buffer.dones

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.hyperparams.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + self.hyperparams.gamma * self.hyperparams.gae_lambda * (1 - dones[t]) * gae
            advantages.append(gae)

        # need to reverse because we're working back to front
        advantages.reverse()

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = advantages + torch.FloatTensor(self.buffer.values).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train(self, last_value):
        """
        Core PPO update — runs multiple epochs over the collected rollout

        last value represents the value of the state after the last action is taken, as it is needed
        to know the value of the state following our final action as per the PPO training algorithm

        Returns:
            Dict of mean minibatch metrics (empty dict if there was nothing to train).
        """
        if len(self.buffer.rewards) == 0:
            return {}

        advantages, returns = self.compute_advantages(last_value)

        # Convert rollout lists to contiguous arrays first to avoid slow
        # tensor-from-list-of-ndarrays construction.
        states_np = np.asarray(self.buffer.states, dtype=np.float32)
        masks_np = np.asarray(self.buffer.masks, dtype=np.float32)
        actions_np = np.asarray(self.buffer.actions, dtype=np.int64)
        old_log_probs_np = np.asarray(self.buffer.log_probs, dtype=np.float32)

        states = torch.from_numpy(states_np).to(self.device)
        masks = torch.from_numpy(masks_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(self.device)

        total_loss = 0.0
        total_actor = 0.0
        total_critic = 0.0
        total_entropy = 0.0
        n_batches = 0

        # PPO trains multiple epochs on the SAME rollout
        for epoch in range(self.hyperparams.epochs):
            # Mini-batch updates
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.hyperparams.batch_size):
                batch_idx = indices[start:start + self.hyperparams.batch_size]

                b_states    = states[batch_idx]
                b_masks     = masks[batch_idx]
                b_actions   = actions[batch_idx]
                b_old_lp    = old_log_probs[batch_idx]
                b_advantages= advantages[batch_idx]
                b_returns   = returns[batch_idx]

                # Forward pass
                probs, values = self.model(b_states, b_masks)
                dist          = Categorical(probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy       = dist.entropy().mean()

                # --- PPO clipped surrogate (Schulman et al.): always min(r*A, clip(r)*A) ---
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
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic Loss ---
                # Keep both tensors 1D to avoid scalar-vs-vector broadcasting when batch size is 1.
                critic_loss = nn.MSELoss()(values.view(-1), b_returns.view(-1))

                # --- Total Loss ---
                loss = (
                    actor_loss
                    + self.hyperparams.value_coef  * critic_loss
                    - self.hyperparams.entropy_coef * entropy   # minus because we WANT entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams.max_grad_norm)  # gradient clipping
                self.optimizer.step()

                total_loss += float(loss.detach().item())
                total_actor += float(actor_loss.detach().item())
                total_critic += float(critic_loss.detach().item())
                total_entropy += float(entropy.detach().item())
                n_batches += 1

        # Discard rollout after training
        self.buffer.clear()

        if n_batches == 0:
            return {}
        inv = 1.0 / n_batches
        return {
            "ppo_total_loss": total_loss * inv,
            "ppo_actor_loss": total_actor * inv,
            "ppo_critic_loss": total_critic * inv,
            "ppo_entropy_mean": total_entropy * inv,
        }


    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

    def save(self, path):
        torch.save(self.model.state_dict(), path)