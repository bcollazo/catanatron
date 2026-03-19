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
import csv
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from CapstoneAgent import CapstoneAgent
from PlacementAgent import PlacementAgent, RandomPlacementAgent, make_placement_agent
from AgentRouter import AgentRouter
from action_map import validate as validate_action_mapping, describe_action
from device import get_device

import torch
import numpy as np
import gymnasium
import catanatron.gym
from catanatron.json import GameEncoder
from catanatron.models.player import Color


def _timestamped_path(path: str) -> str:
    """Insert a UTC timestamp before the file extension.

    ``capstone_model.pt`` -> ``capstone_model_20260317T1423Z.pt``
    """
    root, ext = os.path.splitext(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"{root}_{stamp}{ext}"


OBS_SIZE = 1258
HIDDEN_SIZE = 512
PLACEMENT_HIDDEN_SIZE = 256
MAX_STEPS_PER_GAME = 5000
DEFAULT_MODEL_PATH = "capstone_agent/capstone_model.pt"
DEFAULT_PLACEMENT_MODEL_PATH = "capstone_agent/placement_model.pt"
DEFAULT_BENCHMARK_CSV = "capstone_agent/benchmarks/training_metrics.csv"


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


class BenchmarkLogger:
    DEFAULT_HEADER = [
        "timestamp_utc",
        "run_name",
        "mode",
        "loaded_model_path",
        "game_index",
        "games_total",
        "status",
        "won",
        "terminated",
        "truncated",
        "steps",
        "reward",
        "cum_wins",
        "cum_losses",
        "cum_truncations",
        "cum_win_rate",
        "self_seat",
        "went_first",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            self.header = list(self.DEFAULT_HEADER)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
        else:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None) or []
            # Keep existing files backward-compatible instead of changing their schema mid-run.
            self.header = (
                existing_header
                if existing_header
                else list(self.DEFAULT_HEADER)
            )

    def write_row(self, row: dict):
        filtered = {key: row.get(key, "") for key in self.header}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(filtered)


def _unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def maybe_save_game_json(env, out_dir: Optional[str], game_index: int, every: int):
    if not out_dir:
        return None
    if every <= 0:
        every = 100
    # Save first game of each block: 1, 1+every, 1+2*every, ...
    if (game_index - 1) % every != 0:
        return None

    core_env = _unwrap_env(env)
    game = getattr(core_env, "game", None)
    if game is None:
        return None

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{game.id}.json")
    with open(filepath, "w") as f:
        f.write(json.dumps(game, cls=GameEncoder))
    return filepath


def _self_seat(env, self_color: Color = Color.BLUE) -> Optional[int]:
    core_env = _unwrap_env(env)
    game = getattr(core_env, "game", None)
    if game is None:
        return None
    return game.state.color_to_index.get(self_color)


def simulate_game(
    agent: CapstoneAgent,
    env,
    max_steps: int = MAX_STEPS_PER_GAME,
    verbose: bool = False,
    store_in_buffer: bool = False,
) -> GameResult:
    """Play one full game using the agent and return a GameResult.

    Args:
        agent: A CapstoneAgent, PlacementAgent, or AgentRouter.
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

    # Replay JSON export happens after this function returns. Avoid resetting the
    # env here, otherwise the saved replay may capture a fresh game instead of
    # the one we just finished. Because episodes are terminal at this point,
    # bootstrap value for GAE is 0.
    agent.train(0.0)

    return result


def make_agent_and_env(
    obs_size: int = OBS_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    model_path: Optional[str] = None,
    placement_model_path: Optional[str] = None,
    placement_strategy: str = "model",
):
    """Create a routed agent + env pair.

    Args:
        placement_strategy: ``"model"`` for the learned PlacementAgent,
            ``"random"`` for uniform-random placement.

    Returns an AgentRouter that delegates initial-placement decisions to
    the chosen placement agent and all other decisions to the main
    CapstoneAgent.
    """
    validate_action_mapping()

    main_agent = CapstoneAgent(obs_size=obs_size, hidden_size=hidden_size)
    if model_path is not None:
        main_agent.load(model_path)

    placement_agent = make_placement_agent(
        placement_strategy,
        obs_size=obs_size,
        hidden_size=PLACEMENT_HIDDEN_SIZE,
    )
    if placement_model_path is not None:
        placement_agent.load(placement_model_path)

    env = gymnasium.make("catanatron/CapstoneCatanatron-v0")
    router = AgentRouter(placement_agent, main_agent, env)
    return router, env


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
        "--load", type=str, default=None, help="Path to saved main-agent model weights"
    )
    parser.add_argument(
        "--save", type=str, default=DEFAULT_MODEL_PATH,
        help=(
            "Path to save main-agent model weights after all games. "
            "In --train mode, this path is auto-used for resume if it already exists."
        ),
    )
    parser.add_argument(
        "--placement-strategy", type=str, default="model",
        choices=["model", "random"],
        help="Placement agent strategy: 'model' (learned) or 'random'.",
    )
    parser.add_argument(
        "--placement-model", type=str, default=None,
        help="Path to saved placement-agent model weights (ignored for --placement-strategy random)",
    )
    parser.add_argument(
        "--save-placement-model", type=str, default=DEFAULT_PLACEMENT_MODEL_PATH,
        help="Path to save placement-agent weights after all games.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore existing saved weights and train from scratch.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional benchmark run label. Defaults to timestamp-based name.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default=DEFAULT_BENCHMARK_CSV,
        help="CSV path for per-game benchmark logs.",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Disable benchmark CSV logging.",
    )
    parser.add_argument(
        "--save-games-json-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to save game JSONs (with action_records) for GUI replay."
        ),
    )
    parser.add_argument(
        "--save-games-json-every",
        type=int,
        default=100,
        help=(
            "Save one game JSON every N games (default: 100), "
            "saving the first game of each N-game block."
        ),
    )
    args = parser.parse_args()

    loaded_model_path = args.load
    loaded_placement_path = args.placement_model
    if args.train and not args.fresh_start:
        if loaded_model_path is None and args.save and os.path.exists(args.save):
            loaded_model_path = args.save
            print(f"Resuming training from existing main weights: {loaded_model_path}")
        if (
            args.placement_strategy == "model"
            and loaded_placement_path is None
            and args.save_placement_model
            and os.path.exists(args.save_placement_model)
        ):
            loaded_placement_path = args.save_placement_model
            print(f"Resuming training from existing placement weights: {loaded_placement_path}")

    agent, env = make_agent_and_env(
        model_path=loaded_model_path,
        placement_model_path=loaded_placement_path,
        placement_strategy=args.placement_strategy,
    )
    main_params = sum(p.numel() for p in agent.main_agent.model.parameters())
    pa = agent.placement_agent
    if hasattr(pa, "model"):
        place_desc = f"{sum(p.numel() for p in pa.model.parameters()):,} params"
    else:
        place_desc = args.placement_strategy
    device = get_device()
    print(
        f"Main agent ready      ({main_params:,} params, obs={OBS_SIZE}, actions=245)\n"
        f"Placement agent ready ({place_desc})\n"
        f"Device: {device}"
    )
    print()

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    benchmark = None if args.no_benchmark else BenchmarkLogger(args.benchmark_csv)

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
        saved_game_path = maybe_save_game_json(
            env,
            args.save_games_json_dir,
            g,
            args.save_games_json_every,
        )
        if saved_game_path is not None:
            print(f"  saved replay json: {saved_game_path}")

        if benchmark is not None:
            self_seat = _self_seat(env)
            benchmark.write_row(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "run_name": run_name,
                    "mode": "train" if args.train else "eval",
                    "loaded_model_path": loaded_model_path or "",
                    "game_index": g,
                    "games_total": args.games,
                    "status": status,
                    "won": int(result.won),
                    "terminated": int(result.terminated),
                    "truncated": int(result.truncated),
                    "steps": result.steps,
                    "reward": float(result.cumulative_reward),
                    "cum_wins": wins,
                    "cum_losses": losses,
                    "cum_truncations": truncations,
                    "cum_win_rate": float(wins / g),
                    "self_seat": self_seat if self_seat is not None else "",
                    "went_first": (
                        int(self_seat == 0) if self_seat is not None else ""
                    ),
                }
            )

    print()
    print(f"Results: {wins}W / {losses}L / {truncations}T  ({args.games} games)")
    if benchmark is not None:
        print(f"Benchmarks logged to: {args.benchmark_csv} (run_name={run_name})")

    if args.train and args.save:
        placement_save = (
            args.save_placement_model
            if args.placement_strategy == "model"
            else None
        )
        main_ckpt = _timestamped_path(args.save)
        place_ckpt = _timestamped_path(placement_save) if placement_save else None

        agent.save(args.save, placement_save)
        agent.save(main_ckpt, place_ckpt)

        print(f"Main model saved to {args.save}  (checkpoint: {main_ckpt})")
        if placement_save:
            print(f"Placement model saved to {placement_save}  (checkpoint: {place_ckpt})")


if __name__ == "__main__":
    main()
