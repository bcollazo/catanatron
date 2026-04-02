"""Run one or more Capstone agent simulations.

Usage:
    # From the repo root:
    python capstone_agent/run_simulation.py              # 1 game, no training
    python capstone_agent/run_simulation.py --games 10   # 10 games, no training
    python capstone_agent/run_simulation.py --train      # PPO updates by buffered steps
    python capstone_agent/run_simulation.py --train --train-every-steps 4096
                                                      # PPO update every 4096 steps
    python capstone_agent/run_simulation.py --train --train-update-trigger games \
        --train-every-games 20                       # PPO update every 20 games
    python capstone_agent/run_simulation.py --games 10 --enemy alphabeta
                                                      # evaluate/train vs AlphaBeta

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
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from capstone_agent.MainPlayAgent import MainPlayAgent
from PlacementAgent import PlacementAgent, RandomPlacementAgent, make_placement_agent
from CapstoneAgent import CapstoneAgent
from action_map import validate as validate_action_mapping, describe_action
from device import get_device

import torch
import numpy as np
import gymnasium
import catanatron.gym
from catanatron.json import GameEncoder
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from CONSTANTS import (FEATURE_SPACE_SIZE, MAIN_PLAY_AGENT_HIDDEN_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE, 
                       MAX_STEPS_PER_GAME,
                       DEFAULT_BENCHMARK_CSV, DEFAULT_MAIN_PLAY_MODEL_PATH, DEFAULT_PLACEMENT_MODEL_PATH)

from CONFIG import (DEFAULT_TRAIN_UPDATE_STEPS)
def _timestamped_path(path: str) -> str:
    """Insert a UTC timestamp before the file extension.

    ``capstone_model.pt`` -> ``capstone_model_20260317T1423Z.pt``
    """
    root, ext = os.path.splitext(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"{root}_{stamp}{ext}"


@dataclass
class GameResult:
    steps: int = 0
    cumulative_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    won: bool = False
    action_log: List[int] = field(default_factory=list)
    # State after the last env.step (for PPO bootstrap); same as next_obs on final tick.
    bootstrap_obs: Optional[np.ndarray] = None
    bootstrap_mask: Optional[np.ndarray] = None

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


def _ppo_train_with_bootstrap(agent, bootstrap_obs, bootstrap_mask) -> None:
    """CapstoneAgent.train expects (obs, mask); standalone agents use scalar bootstrap."""
    if hasattr(agent, "placement_agent") and hasattr(agent, "main_agent"):
        agent.train(bootstrap_obs, bootstrap_mask)
        return
    if bootstrap_obs is None or bootstrap_mask is None:
        agent.train(0.0)
        return
    device = get_device()
    with torch.no_grad():
        obs_t = torch.FloatTensor(bootstrap_obs).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(bootstrap_mask).unsqueeze(0).to(device)
        _, last_v = agent.model(obs_t, mask_t)
    agent.train(last_v.item())


def _buffer_transition_count(agent) -> int:
    """Return total rollout transitions currently buffered on an agent/router."""
    total = 0
    if hasattr(agent, "buffer"):
        total += len(agent.buffer.rewards)
    if hasattr(agent, "main_agent") and hasattr(agent.main_agent, "buffer"):
        total += len(agent.main_agent.buffer.rewards)
    if hasattr(agent, "placement_agent") and hasattr(agent.placement_agent, "buffer"):
        total += len(agent.placement_agent.buffer.rewards)
    return total


def make_enemy_player(
    enemy_type: str,
    color: Color = Color.RED,
    alphabeta_depth: int = 2,
    alphabeta_prunning: bool = False,
):
    """Construct the training/eval opponent used by CapstoneCatanatronEnv."""
    if enemy_type == "random":
        return RandomPlayer(color)
    if enemy_type == "alphabeta":
        return AlphaBetaPlayer(
            color, depth=alphabeta_depth, prunning=alphabeta_prunning
        )
    if enemy_type == "alphabeta-prune":
        return AlphaBetaPlayer(color, depth=alphabeta_depth, prunning=True)
    if enemy_type == "same-turn-ab":
        return SameTurnAlphaBetaPlayer(color, depth=alphabeta_depth)
    if enemy_type == "value":
        return ValueFunctionPlayer(color)
    if enemy_type == "vp":
        return VictoryPointPlayer(color)
    if enemy_type == "weighted":
        return WeightedRandomPlayer(color)
    raise ValueError(f"Unknown enemy type: {enemy_type}")


def resolve_map_template(map_template: str, map_mode: str) -> str:
    """Resolve AUTO template defaults based on selected map mode."""
    if map_template != "AUTO":
        return map_template
    return "TOURNAMENT" if map_mode == "fixed" else "BASE"


def simulate_game(
    agent,
    env,
    max_steps: int = MAX_STEPS_PER_GAME,
    verbose: bool = False,
    store_in_buffer: bool = False,
) -> GameResult:
    """Play one full game using the agent and return a GameResult.

    Args:
        agent: A CapstoneAgent, PlacementAgent, or MainPlayAgent
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
            agent.store(
                obs, mask, action, log_prob, reward, value, done, next_obs=next_obs
            )

        result.bootstrap_obs = next_obs
        result.bootstrap_mask = next_mask

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
    agent,
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
    # the one we just finished.
    _ppo_train_with_bootstrap(agent, result.bootstrap_obs, result.bootstrap_mask)

    return result


def make_agent_and_env(
    obs_size: int = FEATURE_SPACE_SIZE,
    hidden_size: int = MAIN_PLAY_AGENT_HIDDEN_SIZE,
    model_path: Optional[str] = None,
    placement_model_path: Optional[str] = None,
    placement_strategy: str = "model",
    enemy_type: str = "random",
    enemy_ab_depth: int = 2,
    enemy_ab_prunning: bool = False,
    map_template: str = "BASE",
    map_mode: str = "fixed",
    fixed_map_seed: int = 0,
):
    """Create a routed agent + env pair.

    Args:
        placement_strategy: ``"model"`` for the learned PlacementAgent,
            ``"random"`` for uniform-random placement.
        enemy_type: Opponent bot for all non-blue turns in environment.
        map_template: Board template ("BASE", "MINI", or "TOURNAMENT").
        map_mode: "fixed" for deterministic map layout, "random" for reshuffled map.
        fixed_map_seed: Seed used when map_mode is "fixed".

    Returns a CapstoneAgent that delegates initial-placement decisions to
    the chosen placement agent and all other decisions to the main
    MainPlayAgent.
    """
    validate_action_mapping()

    main_agent = MainPlayAgent(obs_size=obs_size, hidden_size=hidden_size)
    if model_path is not None:
        main_agent.load(model_path)

    placement_agent = make_placement_agent(
        placement_strategy,
        obs_size=obs_size,
        hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
    )
    if placement_model_path is not None:
        placement_agent.load(placement_model_path)

    enemy = make_enemy_player(
        enemy_type,
        color=Color.RED,
        alphabeta_depth=enemy_ab_depth,
        alphabeta_prunning=enemy_ab_prunning,
    )
    randomize_map = map_mode == "random" and map_template != "TOURNAMENT"
    env = gymnasium.make(
        "catanatron/CapstoneCatanatron-v0",
        config={
            "enemies": [enemy],
            "map_type": map_template,
            "randomize_map": randomize_map,
            "fixed_map_seed": fixed_map_seed,
        },
    )
    router = CapstoneAgent(placement_agent, main_agent)

    return router, env


def main():
    parser = argparse.ArgumentParser(description="Run Capstone agent simulations")
    parser.add_argument(
        "--games", type=int, default=1, help="Number of games to simulate"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable PPO training updates (configured by trigger/interval args).",
    )
    parser.add_argument(
        "--train-update-trigger",
        type=str,
        choices=["steps", "games"],
        default="steps",
        help=(
            "What controls PPO update timing in --train mode: "
            "'steps' (default) or 'games'."
        ),
    )
    parser.add_argument(
        "--train-every-steps",
        type=int,
        default=DEFAULT_TRAIN_UPDATE_STEPS,
        help=(
            "When --train-update-trigger=steps, run PPO every N buffered "
            f"transitions (default: {DEFAULT_TRAIN_UPDATE_STEPS})."
        ),
    )
    parser.add_argument(
        "--train-every-games",
        type=int,
        default=20,
        help=(
            "When --train-update-trigger=games, run PPO every N completed games "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-step action log"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Path to saved main-agent model weights"
    )
    parser.add_argument(
        "--save", type=str, default=DEFAULT_MAIN_PLAY_MODEL_PATH,
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
        "--enemy",
        type=str,
        default="random",
        choices=[
            "random",
            "alphabeta",
            "alphabeta-prune",
            "same-turn-ab",
            "value",
            "vp",
            "weighted",
        ],
        help=(
            "Opponent bot in env (controls non-blue turns). "
            "Use 'alphabeta' to train/eval directly vs AlphaBeta."
        ),
    )
    parser.add_argument(
        "--enemy-ab-depth",
        type=int,
        default=2,
        help="AlphaBeta depth when --enemy is alphabeta/alphabeta-prune/same-turn-ab.",
    )
    parser.add_argument(
        "--enemy-ab-prunning",
        action="store_true",
        help="Enable AlphaBeta pruning when --enemy=alphabeta.",
    )
    parser.add_argument(
        "--map-template",
        type=str,
        default="AUTO",
        choices=["AUTO", "BASE", "MINI", "TOURNAMENT"],
        help=(
            "Board template for simulation games. "
            "AUTO uses TOURNAMENT for fixed mode and BASE for random mode."
        ),
    )
    parser.add_argument(
        "--map-mode",
        type=str,
        default="fixed",
        choices=["fixed", "random"],
        help=(
            "Map layout mode: 'fixed' (default, deterministic by seed) "
            "or 'random' (reshuffle each game)."
        ),
    )
    parser.add_argument(
        "--fixed-map-seed",
        type=int,
        default=0,
        help="Seed used to generate deterministic map layout when --map-mode=fixed.",
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
    if args.train_every_steps < 1:
        parser.error("--train-every-steps must be >= 1")
    if args.train_every_games < 1:
        parser.error("--train-every-games must be >= 1")
    if args.enemy_ab_depth < 1:
        parser.error("--enemy-ab-depth must be >= 1")
    if args.fixed_map_seed < 0:
        parser.error("--fixed-map-seed must be >= 0")
    resolved_map_template = resolve_map_template(args.map_template, args.map_mode)

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
        enemy_type=args.enemy,
        enemy_ab_depth=args.enemy_ab_depth,
        enemy_ab_prunning=args.enemy_ab_prunning,
        map_template=resolved_map_template,
        map_mode=args.map_mode,
        fixed_map_seed=args.fixed_map_seed,
    )
    main_params = sum(p.numel() for p in agent.main_agent.model.parameters())
    pa = agent.placement_agent
    if hasattr(pa, "model"):
        place_desc = f"{sum(p.numel() for p in pa.model.parameters()):,} params"
    else:
        place_desc = args.placement_strategy
    device = get_device()

    print(
        f"Main agent ready      ({main_params:,} params, obs={FEATURE_SPACE_SIZE}, actions=245)\n"
        f"Placement agent ready ({place_desc})\n"
        f"Device: {device}"
    )

    enemy_detail = args.enemy
    if args.enemy in {"alphabeta", "alphabeta-prune", "same-turn-ab"}:
        enemy_detail += f" (depth={args.enemy_ab_depth}"
        if args.enemy == "alphabeta":
            enemy_detail += f", prunning={args.enemy_ab_prunning}"
        enemy_detail += ")"
    map_detail = f"{resolved_map_template} / {args.map_mode}"
    if args.map_template == "AUTO":
        map_detail += " (auto-selected)"
    if args.map_mode == "fixed":
        map_detail += f" (seed={args.fixed_map_seed})"
    if resolved_map_template == "TOURNAMENT" and args.map_mode == "random":
        map_detail += " [TOURNAMENT map is always fixed]"
    training_detail = "disabled"

    if args.train:
        if args.train_update_trigger == "steps":
            training_detail = (
                f"enabled, trigger=steps, every {args.train_every_steps} transitions"
            )
        else:
            training_detail = (
                f"enabled, trigger=games, every {args.train_every_games} games"
            )
    replay_detail = "disabled"
    if args.save_games_json_dir:
        replay_detail = (
            f"enabled, dir={args.save_games_json_dir}, every {args.save_games_json_every} games"
        )
    print(
        "Run configuration:\n"
        f"  Main agent: {main_params:,} params (obs={FEATURE_SPACE_SIZE}, actions=245)\n"
        f"  Main weights: {loaded_model_path or '[fresh/random init]'}\n"
        f"  Placement agent: strategy={args.placement_strategy}, {place_desc}\n"
        f"  Placement weights: {loaded_placement_path or '[none loaded]'}\n"
        f"  Opponent: {enemy_detail}\n"
        f"  Map: {map_detail}\n"
        f"  Training: {training_detail}\n"
        f"  Replay JSON save: {replay_detail}\n"
        f"  Benchmark CSV: {'disabled' if args.no_benchmark else args.benchmark_csv}\n"
        f"  Device: {device}"
    )
    print()

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    benchmark = None if args.no_benchmark else BenchmarkLogger(args.benchmark_csv)

    wins, losses, truncations = 0, 0, 0
    games_since_update = 0
    recent_wins = deque(maxlen=300)
    last_bootstrap_obs: Optional[np.ndarray] = None
    last_bootstrap_mask: Optional[np.ndarray] = None

    for g in range(1, args.games + 1):
        if args.train:
            result = simulate_game(
                agent, env, verbose=args.verbose, store_in_buffer=True
            )
            last_bootstrap_obs = result.bootstrap_obs
            last_bootstrap_mask = result.bootstrap_mask
            games_since_update += 1
        else:
            result = simulate_game(agent, env, verbose=args.verbose)

        wins += result.won
        losses += result.terminated and not result.won
        truncations += result.truncated
        recent_wins.append(int(result.won))
        rolling_n = len(recent_wins)
        rolling_300_win_rate = sum(recent_wins) / rolling_n if rolling_n > 0 else 0.0

        status = "WON" if result.won else ("TRUNCATED" if result.truncated else "LOST")
        print(
            f"Game {g:4d}/{args.games}:  {status:>9s}  "
            f"steps={result.steps:4d}  reward={result.cumulative_reward:+.1f}  "
            f"rolling_300_win_rate={rolling_300_win_rate:.1%} (n={rolling_n})"
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

        if args.train:
            buffered = _buffer_transition_count(agent)
            should_update = False
            update_reason = ""

            if args.train_update_trigger == "steps":
                if buffered >= args.train_every_steps:
                    should_update = True
                    update_reason = f"buffered transitions reached {buffered}"
            else:
                if games_since_update >= args.train_every_games:
                    should_update = True
                    update_reason = f"completed games in batch reached {games_since_update}"

            if should_update and buffered > 0:
                batch_games = games_since_update
                _ppo_train_with_bootstrap(
                    agent, last_bootstrap_obs, last_bootstrap_mask
                )
                print(
                    f"  trained PPO on {buffered} buffered transitions "
                    f"(games in batch: {batch_games}; trigger={args.train_update_trigger}: {update_reason})"
                )
                games_since_update = 0

    if args.train:
        buffered = _buffer_transition_count(agent)
        if buffered > 0:
            batch_games = games_since_update
            _ppo_train_with_bootstrap(
                agent, last_bootstrap_obs, last_bootstrap_mask
            )
            print(
                f"Final PPO flush on {buffered} buffered transitions "
                f"(games in batch: {batch_games})"
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
