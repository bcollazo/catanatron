"""Run one or more Capstone agent simulations.

Usage:
    # From the repo root:
    python capstone_agent/run_simulation.py              # 1 game, no training
    python capstone_agent/run_simulation.py --games 10   # 10 games, no training
    python capstone_agent/run_simulation.py --train      # 1 game + PPO update

    # Or import and call from your own code:
    from run_simulation import simulate_game, simulate_and_train
    result = simulate_game(agent, env)
    result = simulate_and_train(agent, env)
"""

import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from CapstoneAgent import CapstoneAgent
from action_map import validate as validate_action_mapping, describe_action

import torch
import numpy as np
import gymnasium
import catanatron.gym


OBS_SIZE = 1258
HIDDEN_SIZE = 512
MAX_STEPS_PER_GAME = 5000


@dataclass
class GameResult:
    steps: int = 0
    cumulative_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    won: bool = False
    action_log: List[int] = field(default_factory=list)

    @property
    def done(self):
        return self.terminated or self.truncated


def simulate_game(
    agent: CapstoneAgent,
    env,
    max_steps: int = MAX_STEPS_PER_GAME,
    verbose: bool = False,
    store_in_buffer: bool = False,
) -> GameResult:
    """Play one full game using the agent and return a GameResult.

    Args:
        agent: The CapstoneAgent to use for action selection.
        env: A CapstoneCatanatronEnv gymnasium environment.
        max_steps: Safety limit on the number of steps.
        verbose: Print per-step action descriptions.
        store_in_buffer: If True, store transitions in the agent's rollout
            buffer (needed if you plan to call agent.train() afterward).

    Returns:
        A GameResult with stats about the game.
    """
    obs, info = env.reset()
    mask = info["action_mask"]
    result = GameResult()

    for step in range(1, max_steps + 1):
        action, log_prob, value = agent.select_action(obs, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["action_mask"]

        if store_in_buffer:
            agent.store(obs, mask, action, log_prob, reward, value, done)

        result.steps = step
        result.cumulative_reward += reward
        result.action_log.append(action)

        if verbose and (step <= 5 or step % 50 == 0 or done):
            desc = describe_action(action)
            print(
                f"  Step {step:4d}: action={action:3d} ({desc})  "
                f"reward={reward:+.1f}  value_est={value:+.4f}"
            )

        if done:
            result.terminated = terminated
            result.truncated = truncated
            result.won = reward > 0
            break

        obs, mask = next_obs, next_mask

    return result


def simulate_and_train(
    agent: CapstoneAgent,
    env,
    max_steps: int = MAX_STEPS_PER_GAME,
    verbose: bool = False,
) -> GameResult:
    """Play one game, then run a PPO update on the collected rollout."""
    result = simulate_game(
        agent, env, max_steps=max_steps, verbose=verbose, store_in_buffer=True
    )

    obs, info = env.reset()
    mask = info["action_mask"]
    with torch.no_grad():
        _, last_value = agent.model(
            torch.FloatTensor(obs).unsqueeze(0),
            torch.FloatTensor(mask).unsqueeze(0),
        )
    agent.train(last_value.item())

    return result


def make_agent_and_env(
    obs_size: int = OBS_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    model_path: Optional[str] = None,
):
    """Create a fresh agent + env pair. Optionally load saved weights."""
    validate_action_mapping()
    agent = CapstoneAgent(obs_size=obs_size, hidden_size=hidden_size)
    if model_path is not None:
        agent.load(model_path)
    env = gymnasium.make("catanatron/CapstoneCatanatron-v0")
    return agent, env


def main():
    parser = argparse.ArgumentParser(description="Run Capstone agent simulations")
    parser.add_argument(
        "--games", type=int, default=1, help="Number of games to simulate"
    )
    parser.add_argument(
        "--train", action="store_true", help="Run a PPO update after each game"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-step action log"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Path to saved model weights"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save model weights after all games (only with --train)",
    )
    args = parser.parse_args()

    agent, env = make_agent_and_env(model_path=args.load)
    params = sum(p.numel() for p in agent.model.parameters())
    print(f"Agent ready  ({params:,} params, obs={OBS_SIZE}, actions=245)")
    print()

    wins, losses, truncations = 0, 0, 0

    for g in range(1, args.games + 1):
        if args.train:
            result = simulate_and_train(agent, env, verbose=args.verbose)
        else:
            result = simulate_game(agent, env, verbose=args.verbose)

        wins += result.won
        losses += result.terminated and not result.won
        truncations += result.truncated

        status = "WON" if result.won else ("TRUNCATED" if result.truncated else "LOST")
        print(
            f"Game {g:4d}/{args.games}:  {status:>9s}  "
            f"steps={result.steps:4d}  reward={result.cumulative_reward:+.1f}"
        )

    print()
    print(f"Results: {wins}W / {losses}L / {truncations}T  ({args.games} games)")

    if args.train and args.save:
        agent.save(args.save)
        print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
