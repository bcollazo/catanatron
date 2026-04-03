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
import shutil
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from MainPlayAgent import MainPlayAgent
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
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.rl_capstone_agent import RLCapstonePlayer

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


def _ensure_parent_dir(path: Optional[str]):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _champion_history_paths(
    history_dir: str,
    promotion_index: int,
    game_index: int,
    eval_win_rate: float,
) -> Tuple[str, str]:
    """Unique archive paths for a promoted champion (main + placement)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    wr_tag = f"{eval_win_rate:.4f}".replace(".", "p")
    stem = f"promo{promotion_index:04d}_game{game_index}_wr{wr_tag}_{stamp}"
    main_p = os.path.join(history_dir, f"champion_main_{stem}.pt")
    place_p = os.path.join(history_dir, f"champion_placement_{stem}.pt")
    return main_p, place_p


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


@dataclass
class ScheduledEnemyPhase:
    enemy_type: str
    games: int
    enemy_ab_depth: Optional[int] = None


@dataclass(frozen=True)
class EnemySpec:
    enemy_type: str
    enemy_ab_depth: Optional[int] = None


PRESET_DEFAULTS = {
    "games": 1,
    "train": False,
    "placement_strategy": "model",
    "placement_model": None,
    "save_every_games": 0,
    "enemy_fixed_schedule": False,
    "enemy_schedule": "weighted:50000,value:50000,alphabeta@1:50000,alphabeta@2:50000",
    "enemy_smooth_mix": False,
    "enemy_mix_start": "weighted:1.0",
    "enemy_mix_end": "value:1.0",
    "enemy_mix_games": 50000,
    "enemy_mix_seed": 0,
    "enemy": "random",
    "enemy_ab_depth": 2,
    "enemy_mcts_n": 100,
    "enemy_greedy_n": 25,
    "map_mode": "fixed",
    "self_play_ladder": False,
    "self_play_eval_every_games": 1000,
    "self_play_eval_games": 400,
    "self_play_promotion_threshold": 0.55,
    "enemy_switch_log_every": 0,
    "per_enemy_min_samples": 20,
}


def _set_if_default(args, key: str, value):
    if getattr(args, key) == PRESET_DEFAULTS[key]:
        setattr(args, key, value)


def apply_training_preset(args):
    """Apply high-level preset defaults while allowing manual overrides."""
    if args.preset == "none":
        return

    diff = args.difficulty
    _set_if_default(args, "train", True)
    _set_if_default(args, "map_mode", "fixed")

    # Safer placement baseline for training presets unless user overrides.
    _set_if_default(args, "placement_strategy", "alphabeta")
    _set_if_default(args, "placement_model", None)

    if args.preset == "warmup":
        games_by_diff = {"easy": 50_000, "medium": 150_000, "hard": 300_000}
        _set_if_default(args, "games", games_by_diff[diff])
        _set_if_default(args, "enemy", "weighted")
        _set_if_default(args, "save_every_games", 1000)
        return

    if args.preset == "blend":
        games_by_diff = {"easy": 80_000, "medium": 200_000, "hard": 400_000}
        mix_games_by_diff = {"easy": 40_000, "medium": 100_000, "hard": 200_000}
        _set_if_default(args, "games", games_by_diff[diff])
        _set_if_default(args, "enemy_smooth_mix", True)
        _set_if_default(args, "enemy_mix_start", "weighted:0.8,value:0.2")
        _set_if_default(args, "enemy_mix_end", "weighted:0.2,value:0.8")
        _set_if_default(args, "enemy_mix_games", mix_games_by_diff[diff])
        _set_if_default(args, "save_every_games", 1000)
        return

    if args.preset == "ab-ramp":
        schedule_by_diff = {
            "easy": "weighted:20000,value:20000,alphabeta@1:20000,alphabeta@2:20000",
            "medium": "weighted:50000,value:50000,alphabeta@1:50000,alphabeta@2:50000",
            "hard": "weighted:100000,value:100000,alphabeta@1:100000,alphabeta@2:100000",
        }
        games_by_diff = {"easy": 80_000, "medium": 200_000, "hard": 400_000}
        _set_if_default(args, "games", games_by_diff[diff])
        _set_if_default(args, "enemy_fixed_schedule", True)
        _set_if_default(args, "enemy_schedule", schedule_by_diff[diff])
        _set_if_default(args, "save_every_games", 1000)
        return

    if args.preset == "self-play":
        games_by_diff = {"easy": 50_000, "medium": 150_000, "hard": 300_000}
        eval_every_by_diff = {"easy": 500, "medium": 1000, "hard": 1500}
        eval_games_by_diff = {"easy": 200, "medium": 400, "hard": 600}
        _set_if_default(args, "games", games_by_diff[diff])
        _set_if_default(args, "self_play_ladder", True)
        _set_if_default(args, "placement_strategy", "model")
        _set_if_default(args, "self_play_eval_every_games", eval_every_by_diff[diff])
        _set_if_default(args, "self_play_eval_games", eval_games_by_diff[diff])
        _set_if_default(args, "save_every_games", 1000)
        return


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


def _action_context(env):
    core = _unwrap_env(env)
    game = getattr(core, "game", None)
    playable_actions = getattr(game, "playable_actions", None) if game is not None else None
    return game, playable_actions


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
    mcts_simulations: int = 100,
    greedy_playouts: int = 25,
    alphabeta_prunning: bool = False,
    main_model_path: Optional[str] = None,
    placement_model_path: Optional[str] = None,
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
    if enemy_type == "mcts":
        return MCTSPlayer(color, num_simulations=mcts_simulations)
    if enemy_type == "greedy":
        return GreedyPlayoutsPlayer(color, num_playouts=greedy_playouts)
    if enemy_type == "rl-capstone":
        if main_model_path is None:
            raise ValueError("main_model_path is required for enemy_type='rl-capstone'")
        return RLCapstonePlayer(
            color,
            settlement_play_load_file=placement_model_path,
            main_play_load_file=main_model_path,
        )
    raise ValueError(f"Unknown enemy type: {enemy_type}")


def resolve_map_template(map_template: str, map_mode: str) -> str:
    """Resolve AUTO template defaults based on selected map mode."""
    if map_template != "AUTO":
        return map_template
    return "TOURNAMENT" if map_mode == "fixed" else "BASE"


def make_env(
    enemy_type: str,
    enemy_ab_depth: int,
    enemy_mcts_n: int,
    enemy_greedy_n: int,
    enemy_ab_prunning: bool,
    map_template: str,
    map_mode: str,
    fixed_map_seed: int,
    enemy_main_model_path: Optional[str] = None,
    enemy_placement_model_path: Optional[str] = None,
):
    enemy = make_enemy_player(
        enemy_type,
        color=Color.RED,
        alphabeta_depth=enemy_ab_depth,
        mcts_simulations=enemy_mcts_n,
        greedy_playouts=enemy_greedy_n,
        alphabeta_prunning=enemy_ab_prunning,
        main_model_path=enemy_main_model_path,
        placement_model_path=enemy_placement_model_path,
    )
    randomize_map = map_mode == "random" and map_template != "TOURNAMENT"
    return gymnasium.make(
        "catanatron/CapstoneCatanatron-v0",
        config={
            "enemies": [enemy],
            "map_type": map_template,
            "randomize_map": randomize_map,
            "fixed_map_seed": fixed_map_seed,
        },
    )


ALLOWED_ENEMY_TYPES = {
    "random",
    "alphabeta",
    "alphabeta-prune",
    "same-turn-ab",
    "value",
    "vp",
    "weighted",
    "mcts",
    "greedy",
    "rl-capstone",
}


def _parse_enemy_spec(spec: str) -> EnemySpec:
    token = spec.strip()
    if not token:
        raise ValueError("empty enemy spec")
    if "@" in token:
        enemy_type, depth_str = token.split("@", 1)
        enemy_type = enemy_type.strip()
        depth = int(depth_str)
        if depth <= 0:
            raise ValueError(f"invalid depth in enemy spec '{spec}'")
        if enemy_type not in ALLOWED_ENEMY_TYPES:
            raise ValueError(f"unknown enemy type '{enemy_type}' in spec '{spec}'")
        return EnemySpec(enemy_type=enemy_type, enemy_ab_depth=depth)
    if token not in ALLOWED_ENEMY_TYPES:
        raise ValueError(f"unknown enemy type '{token}' in spec '{spec}'")
    return EnemySpec(enemy_type=token, enemy_ab_depth=None)


def _enemy_spec_label(enemy: EnemySpec, default_ab_depth: int) -> str:
    if enemy.enemy_type in {"alphabeta", "alphabeta-prune", "same-turn-ab"}:
        depth = enemy.enemy_ab_depth if enemy.enemy_ab_depth is not None else default_ab_depth
        return f"{enemy.enemy_type}@{depth}"
    if enemy.enemy_type in {"mcts", "greedy"} and enemy.enemy_ab_depth is not None:
        return f"{enemy.enemy_type}@{enemy.enemy_ab_depth}"
    return enemy.enemy_type


def _resolved_enemy_param(enemy_type: str, spec_param: Optional[int], args) -> int:
    """Resolve optional @param against per-enemy CLI defaults."""
    if spec_param is not None:
        return spec_param
    if enemy_type in {"alphabeta", "alphabeta-prune", "same-turn-ab"}:
        return args.enemy_ab_depth
    if enemy_type == "mcts":
        return args.enemy_mcts_n
    if enemy_type == "greedy":
        return args.enemy_greedy_n
    return args.enemy_ab_depth


def parse_enemy_mix(mix: str) -> Dict[EnemySpec, float]:
    """Parse enemy mix string into normalized probability weights.

    Format:
      "<enemy>:<prob>,<enemy>@<ab_depth>:<prob>,..."
    Example:
      "weighted:0.8,value:0.2"
    """
    if not mix.strip():
        raise ValueError("enemy mix is empty")
    weights: Dict[EnemySpec, float] = {}
    for raw_token in mix.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid mix token '{token}'. Expected '<enemy>[:depth]:<prob>' format."
            )
        enemy_part, prob_part = token.rsplit(":", 1)
        enemy = _parse_enemy_spec(enemy_part)
        prob = float(prob_part)
        if prob < 0:
            raise ValueError(f"Negative probability in token '{token}'")
        weights[enemy] = weights.get(enemy, 0.0) + prob
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("enemy mix probabilities sum to zero")
    return {enemy: (w / total) for enemy, w in weights.items() if w > 0}


def interpolate_enemy_mix(
    start_mix: Dict[EnemySpec, float],
    end_mix: Dict[EnemySpec, float],
    game_index_1_based: int,
    blend_games: int,
) -> Dict[EnemySpec, float]:
    """Linearly interpolate start->end mix over blend_games."""
    if blend_games <= 1:
        t = 1.0
    else:
        t = min(max((game_index_1_based - 1) / (blend_games - 1), 0.0), 1.0)
    keys = set(start_mix.keys()) | set(end_mix.keys())
    mix: Dict[EnemySpec, float] = {}
    for key in keys:
        s = start_mix.get(key, 0.0)
        e = end_mix.get(key, 0.0)
        w = (1.0 - t) * s + t * e
        if w > 0:
            mix[key] = w
    total = sum(mix.values())
    if total <= 0:
        # Should not happen unless both mixes are empty.
        return dict(end_mix)
    return {k: v / total for k, v in mix.items()}


def sample_enemy_from_mix(
    mix: Dict[EnemySpec, float], rng: np.random.Generator
) -> EnemySpec:
    r = float(rng.random())
    cumulative = 0.0
    last_key = None
    for key in sorted(mix.keys(), key=lambda e: (e.enemy_type, e.enemy_ab_depth or -1)):
        cumulative += mix[key]
        last_key = key
        if r <= cumulative:
            return key
    # Numerical edge-case fallback.
    return last_key if last_key is not None else EnemySpec("random", None)


def dominant_enemy_from_mix(mix: Dict[EnemySpec, float]) -> EnemySpec:
    if not mix:
        return EnemySpec("random", None)
    return max(mix.items(), key=lambda kv: kv[1])[0]


def parse_enemy_schedule(schedule: str) -> list[ScheduledEnemyPhase]:
    """Parse fixed enemy schedule string.

    Format:
      "<enemy>[:<games>],<enemy>@<param>:<games>,..."
    Example:
      "weighted:50000,value:50000,alphabeta@1:50000,alphabeta@2:100000"
    """
    phases: list[ScheduledEnemyPhase] = []
    if not schedule.strip():
        raise ValueError("enemy schedule is empty")

    for raw_token in schedule.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid schedule token '{token}'. Expected '<enemy>[:param]:<games>' format."
            )
        enemy_spec, games_str = token.rsplit(":", 1)
        games = int(games_str)
        if games <= 0:
            raise ValueError(f"Invalid games count in schedule token '{token}'")

        ab_depth = None
        if "@" in enemy_spec:
            enemy_type, depth_str = enemy_spec.split("@", 1)
            enemy_type = enemy_type.strip()
            ab_depth = int(depth_str)
            if ab_depth <= 0:
                raise ValueError(f"Invalid @param in schedule token '{token}'")
        else:
            enemy_type = enemy_spec.strip()
        if enemy_type not in ALLOWED_ENEMY_TYPES:
            raise ValueError(f"Unknown enemy type '{enemy_type}' in schedule token '{token}'")
        phases.append(
            ScheduledEnemyPhase(
                enemy_type=enemy_type,
                games=games,
                enemy_ab_depth=ab_depth,
            )
        )

    if not phases:
        raise ValueError("enemy schedule is empty after parsing")
    return phases


def schedule_phase_for_game(
    game_index_1_based: int, phases: list[ScheduledEnemyPhase]
) -> tuple[int, ScheduledEnemyPhase]:
    """Return (phase_index, phase) for current game index.

    If game index exceeds scheduled total, continue using the final phase.
    """
    remaining = game_index_1_based
    for idx, phase in enumerate(phases):
        if remaining <= phase.games:
            return idx, phase
        remaining -= phase.games
    return len(phases) - 1, phases[-1]


def clear_agent_buffers(agent):
    if hasattr(agent, "main_agent") and hasattr(agent.main_agent, "buffer"):
        agent.main_agent.buffer.clear()
    if hasattr(agent, "placement_agent") and hasattr(agent.placement_agent, "buffer"):
        agent.placement_agent.buffer.clear()


def _get_buffer_lengths(buffer):
    return {
        "states": len(buffer.states),
        "masks": len(buffer.masks),
        "actions": len(buffer.actions),
        "log_probs": len(buffer.log_probs),
        "rewards": len(buffer.rewards),
        "values": len(buffer.values),
        "dones": len(buffer.dones),
    }


def snapshot_agent_buffer_lengths(agent):
    snaps = {}
    if hasattr(agent, "main_agent") and hasattr(agent.main_agent, "buffer"):
        snaps["main"] = _get_buffer_lengths(agent.main_agent.buffer)
    if hasattr(agent, "placement_agent") and hasattr(agent.placement_agent, "buffer"):
        snaps["placement"] = _get_buffer_lengths(agent.placement_agent.buffer)
    return snaps


def _truncate_buffer_to(buffer, snap):
    buffer.states = buffer.states[: snap["states"]]
    buffer.masks = buffer.masks[: snap["masks"]]
    buffer.actions = buffer.actions[: snap["actions"]]
    buffer.log_probs = buffer.log_probs[: snap["log_probs"]]
    buffer.rewards = buffer.rewards[: snap["rewards"]]
    buffer.values = buffer.values[: snap["values"]]
    buffer.dones = buffer.dones[: snap["dones"]]


def truncate_agent_buffers_to_snapshot(agent, snapshots):
    if "main" in snapshots and hasattr(agent, "main_agent"):
        _truncate_buffer_to(agent.main_agent.buffer, snapshots["main"])
    if "placement" in snapshots and hasattr(agent, "placement_agent"):
        _truncate_buffer_to(agent.placement_agent.buffer, snapshots["placement"])


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
        game_ctx, playable_ctx = _action_context(env)
        action, log_prob, value = agent.select_action(
            obs, mask, game=game_ctx, playable_actions=playable_ctx
        )
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
    enemy_mcts_n: int = 100,
    enemy_greedy_n: int = 25,
    enemy_ab_prunning: bool = False,
    map_template: str = "BASE",
    map_mode: str = "fixed",
    fixed_map_seed: int = 0,
    enemy_main_model_path: Optional[str] = None,
    enemy_placement_model_path: Optional[str] = None,
    placement_ab_depth: int = 2,
    placement_ab_prunning: bool = True,
):
    """Create a routed agent + env pair.

    Args:
        placement_strategy: ``"model"`` for the learned PlacementAgent,
            ``"random"`` for uniform-random placement.
        enemy_type: Opponent bot for all non-blue turns in environment.
        map_template: Board template ("BASE", "MINI", or "TOURNAMENT").
        map_mode: "fixed" for deterministic map layout, "random" for reshuffled map.
        fixed_map_seed: Seed used when map_mode is "fixed".
        enemy_main_model_path: Main-play weights for rl-capstone enemy.
        enemy_placement_model_path: Placement weights for rl-capstone enemy.
        placement_ab_depth: AlphaBeta depth for placement strategy ``alphabeta``.
        placement_ab_prunning: AlphaBeta pruning flag for placement strategy ``alphabeta``.

    Returns a CapstoneAgent that delegates initial-placement decisions to
    the chosen placement agent and all other decisions to the main
    MainPlayAgent.
    """
    validate_action_mapping()

    main_agent = MainPlayAgent(obs_size=obs_size, hidden_size=hidden_size)
    if model_path is not None:
        main_agent.load(model_path)

    placement_kwargs = {
        "obs_size": obs_size,
        "hidden_size": PLACEMENT_AGENT_HIDDEN_SIZE,
    }
    if placement_strategy == "alphabeta":
        placement_kwargs["depth"] = placement_ab_depth
        placement_kwargs["prunning"] = placement_ab_prunning
    placement_agent = make_placement_agent(placement_strategy, **placement_kwargs)
    if placement_model_path is not None:
        placement_agent.load(placement_model_path)

    env = make_env(
        enemy_type=enemy_type,
        enemy_ab_depth=enemy_ab_depth,
        enemy_mcts_n=enemy_mcts_n,
        enemy_greedy_n=enemy_greedy_n,
        enemy_ab_prunning=enemy_ab_prunning,
        map_template=map_template,
        map_mode=map_mode,
        fixed_map_seed=fixed_map_seed,
        enemy_main_model_path=enemy_main_model_path,
        enemy_placement_model_path=enemy_placement_model_path,
    )
    router = CapstoneAgent(placement_agent, main_agent)

    return router, env


def evaluate_challenger_vs_champion(
    num_games: int,
    challenger_main_model_path: str,
    challenger_placement_model_path: Optional[str],
    champion_main_model_path: str,
    champion_placement_model_path: Optional[str],
    map_template: str,
    map_mode: str,
    fixed_map_seed: int,
):
    """Seat-balanced evaluation: half games challenger as Blue, half as Red."""
    if num_games <= 0:
        return {"wins": 0, "games": 0, "win_rate": 0.0}

    challenger_blue_games = (num_games + 1) // 2
    champion_blue_games = num_games // 2
    challenger_wins = 0

    # Challenger plays as Blue.
    challenger_agent, challenger_env = make_agent_and_env(
        model_path=challenger_main_model_path,
        placement_model_path=challenger_placement_model_path,
        placement_strategy="model",
        enemy_type="rl-capstone",
        map_template=map_template,
        map_mode=map_mode,
        fixed_map_seed=fixed_map_seed,
        enemy_main_model_path=champion_main_model_path,
        enemy_placement_model_path=champion_placement_model_path,
    )
    for _ in range(challenger_blue_games):
        result = simulate_game(challenger_agent, challenger_env, store_in_buffer=False)
        challenger_wins += int(result.won)

    # Champion plays as Blue. Challenger wins when Blue loses.
    champion_agent, champion_env = make_agent_and_env(
        model_path=champion_main_model_path,
        placement_model_path=champion_placement_model_path,
        placement_strategy="model",
        enemy_type="rl-capstone",
        map_template=map_template,
        map_mode=map_mode,
        fixed_map_seed=fixed_map_seed,
        enemy_main_model_path=challenger_main_model_path,
        enemy_placement_model_path=challenger_placement_model_path,
    )
    for _ in range(champion_blue_games):
        result = simulate_game(champion_agent, champion_env, store_in_buffer=False)
        if result.terminated and not result.won:
            challenger_wins += 1

    return {
        "wins": challenger_wins,
        "games": num_games,
        "win_rate": challenger_wins / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Capstone agent simulations")
    parser.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "warmup", "blend", "ab-ramp", "self-play"],
        help=(
            "High-level run preset. "
            "Use with --difficulty and optionally override any individual flags."
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Preset scale selector for --preset (ignored when --preset=none).",
    )
    parser.add_argument(
        "--print-effective-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the resolved preset/config values after applying overrides.",
    )
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
        choices=["model", "random", "alphabeta"],
        help="Placement agent strategy: 'model' (learned), 'random', or 'alphabeta'.",
    )
    parser.add_argument(
        "--placement-model", type=str, default=None,
        help=(
            "Path to saved placement-agent model weights "
            "(ignored for --placement-strategy random/alphabeta)"
        ),
    )
    parser.add_argument(
        "--save-placement-model", type=str, default=DEFAULT_PLACEMENT_MODEL_PATH,
        help="Path to save placement-agent weights after all games.",
    )
    parser.add_argument(
        "--placement-ab-depth",
        type=int,
        default=2,
        help="AlphaBeta depth when --placement-strategy=alphabeta.",
    )
    parser.add_argument(
        "--placement-ab-prunning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AlphaBeta pruning for --placement-strategy=alphabeta.",
    )
    parser.add_argument(
        "--save-every-games",
        type=int,
        default=0,
        help=(
            "When > 0 in --train mode, periodically overwrite --save "
            "(and --save-placement-model) every N completed games."
        ),
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
            "mcts",
            "greedy",
            "rl-capstone",
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
        "--enemy-mcts-n",
        type=int,
        default=100,
        help="MCTS simulation count for --enemy=mcts and mcts entries in schedule/mix.",
    )
    parser.add_argument(
        "--enemy-greedy-n",
        type=int,
        default=25,
        help="Greedy playout count for --enemy=greedy and greedy entries in schedule/mix.",
    )
    parser.add_argument(
        "--enemy-fixed-schedule",
        action="store_true",
        help=(
            "Use a fixed, phase-based opponent curriculum. "
            "When enabled, --enemy-schedule controls opponent switches by game count."
        ),
    )
    parser.add_argument(
        "--enemy-schedule",
        type=str,
        default="weighted:50000,value:50000,alphabeta@1:50000,alphabeta@2:50000",
        help=(
            "Fixed opponent schedule string used with --enemy-fixed-schedule. "
            "Format: '<enemy>:<games>,<enemy>@<param>:<games>,...'"
        ),
    )
    parser.add_argument(
        "--enemy-smooth-mix",
        action="store_true",
        help=(
            "Enable true probabilistic enemy mixing with smooth (linear) "
            "interpolation from --enemy-mix-start to --enemy-mix-end."
        ),
    )
    parser.add_argument(
        "--enemy-mix-start",
        type=str,
        default="weighted:1.0",
        help=(
            "Starting enemy probability mix. "
            "Format: '<enemy>:<prob>,<enemy>@<param>:<prob>,...'"
        ),
    )
    parser.add_argument(
        "--enemy-mix-end",
        type=str,
        default="value:1.0",
        help=(
            "Ending enemy probability mix. "
            "Format: '<enemy>:<prob>,<enemy>@<param>:<prob>,...'"
        ),
    )
    parser.add_argument(
        "--enemy-mix-games",
        type=int,
        default=50000,
        help="Number of games over which to linearly interpolate start->end mix.",
    )
    parser.add_argument(
        "--enemy-mix-seed",
        type=int,
        default=0,
        help="Random seed used for enemy sampling in smooth-mix mode.",
    )
    parser.add_argument(
        "--enemy-switch-log-every",
        type=int,
        default=0,
        help=(
            "Print enemy-switch debug logs at most once every N games. "
            "0 disables switch logs."
        ),
    )
    parser.add_argument(
        "--per-enemy-min-samples",
        type=int,
        default=20,
        help=(
            "Minimum samples before showing per-enemy rolling win-rate "
            "in per-game logs."
        ),
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
    parser.add_argument(
        "--self-play-ladder",
        action="store_true",
        help=(
            "Enable champion/challenger self-play loop. Challenger trains "
            "against champion and promotes when eval win-rate is high enough."
        ),
    )
    parser.add_argument(
        "--self-play-winner-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In self-play-ladder mode, keep rollout only from challenger wins.",
    )
    parser.add_argument(
        "--champion-main-model",
        type=str,
        default="capstone_agent/models/champion_capstone_model.pt",
        help="Champion main-play model path for self-play-ladder mode.",
    )
    parser.add_argument(
        "--champion-placement-model",
        type=str,
        default="capstone_agent/models/champion_placement_model.pt",
        help="Champion placement model path for self-play-ladder mode.",
    )
    parser.add_argument(
        "--self-play-eval-every-games",
        type=int,
        default=1000,
        help="Run promotion evaluation every N training games in self-play-ladder mode.",
    )
    parser.add_argument(
        "--self-play-eval-games",
        type=int,
        default=400,
        help="Number of head-to-head games per promotion evaluation.",
    )
    parser.add_argument(
        "--self-play-promotion-threshold",
        type=float,
        default=0.55,
        help="Promote challenger when eval win-rate is >= this value.",
    )
    parser.add_argument(
        "--champion-history-dir",
        type=str,
        default=None,
        help=(
            "With --self-play-ladder, each successful promotion also copies the new "
            "champion main (and placement, if used) into this directory. Filenames "
            "include promotion index, training game index, eval win rate, and UTC time."
        ),
    )
    args = parser.parse_args()
    apply_training_preset(args)
    if args.train_every_steps < 1:
        parser.error("--train-every-steps must be >= 1")
    if args.train_every_games < 1:
        parser.error("--train-every-games must be >= 1")
    if args.enemy_ab_depth < 1:
        parser.error("--enemy-ab-depth must be >= 1")
    if args.enemy_mcts_n < 1:
        parser.error("--enemy-mcts-n must be >= 1")
    if args.enemy_greedy_n < 1:
        parser.error("--enemy-greedy-n must be >= 1")
    if args.placement_ab_depth < 1:
        parser.error("--placement-ab-depth must be >= 1")
    if args.save_every_games < 0:
        parser.error("--save-every-games must be >= 0")
    if args.enemy_mix_games < 1:
        parser.error("--enemy-mix-games must be >= 1")
    if args.enemy_mix_seed < 0:
        parser.error("--enemy-mix-seed must be >= 0")
    if args.enemy_switch_log_every < 0:
        parser.error("--enemy-switch-log-every must be >= 0")
    if args.per_enemy_min_samples < 1:
        parser.error("--per-enemy-min-samples must be >= 1")
    if args.fixed_map_seed < 0:
        parser.error("--fixed-map-seed must be >= 0")
    if args.self_play_eval_every_games < 1:
        parser.error("--self-play-eval-every-games must be >= 1")
    if args.self_play_eval_games < 20:
        parser.error("--self-play-eval-games must be >= 20")
    if not (0.0 <= args.self_play_promotion_threshold <= 1.0):
        parser.error("--self-play-promotion-threshold must be between 0 and 1")
    if args.champion_history_dir and not args.self_play_ladder:
        parser.error("--champion-history-dir requires --self-play-ladder")
    schedule_phases: list[ScheduledEnemyPhase] = []
    start_mix: Dict[EnemySpec, float] = {}
    end_mix: Dict[EnemySpec, float] = {}
    mix_rng: Optional[np.random.Generator] = None
    if args.enemy_fixed_schedule:
        try:
            schedule_phases = parse_enemy_schedule(args.enemy_schedule)
        except Exception as exc:
            parser.error(f"--enemy-schedule parse error: {exc}")
        if args.self_play_ladder:
            parser.error("--enemy-fixed-schedule cannot be combined with --self-play-ladder")
    if args.enemy_smooth_mix:
        if args.enemy_fixed_schedule:
            parser.error("--enemy-smooth-mix cannot be combined with --enemy-fixed-schedule")
        if args.self_play_ladder:
            parser.error("--enemy-smooth-mix cannot be combined with --self-play-ladder")
        try:
            start_mix = parse_enemy_mix(args.enemy_mix_start)
            end_mix = parse_enemy_mix(args.enemy_mix_end)
        except Exception as exc:
            parser.error(f"enemy mix parse error: {exc}")
        mix_rng = np.random.default_rng(args.enemy_mix_seed)
    if args.print_effective_config:
        print(
            "Effective config:\n"
            f"  preset={args.preset} difficulty={args.difficulty}\n"
            f"  games={args.games} train={args.train}\n"
            f"  placement_strategy={args.placement_strategy}\n"
            f"  enemy_fixed_schedule={args.enemy_fixed_schedule}\n"
            f"  enemy_smooth_mix={args.enemy_smooth_mix}\n"
        )
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

    first_phase = schedule_phases[0] if schedule_phases else None
    first_mix_enemy = dominant_enemy_from_mix(start_mix) if args.enemy_smooth_mix else None
    agent, env = make_agent_and_env(
        model_path=loaded_model_path,
        placement_model_path=loaded_placement_path,
        placement_strategy=args.placement_strategy,
        enemy_type=(
            first_phase.enemy_type
            if first_phase
            else (first_mix_enemy.enemy_type if first_mix_enemy else args.enemy)
        ),
        enemy_ab_depth=(
            _resolved_enemy_param(first_phase.enemy_type, first_phase.enemy_ab_depth, args)
            if first_phase
            else (
                _resolved_enemy_param(first_mix_enemy.enemy_type, first_mix_enemy.enemy_ab_depth, args)
                if first_mix_enemy
                else _resolved_enemy_param(args.enemy, None, args)
            )
        ),
        enemy_mcts_n=args.enemy_mcts_n,
        enemy_greedy_n=args.enemy_greedy_n,
        enemy_ab_prunning=args.enemy_ab_prunning,
        map_template=resolved_map_template,
        map_mode=args.map_mode,
        fixed_map_seed=args.fixed_map_seed,
        placement_ab_depth=args.placement_ab_depth,
        placement_ab_prunning=args.placement_ab_prunning,
    )
    champion_main_model_path = args.champion_main_model
    champion_placement_model_path = args.champion_placement_model
    if args.self_play_ladder:
        if not args.train:
            parser.error("--self-play-ladder currently requires --train")
        if args.placement_strategy != "model":
            parser.error("--self-play-ladder requires --placement-strategy model")
        if not os.path.exists(champion_main_model_path):
            print(
                "Champion main model missing; bootstrapping from challenger current weights."
            )
            os.makedirs(os.path.dirname(champion_main_model_path), exist_ok=True)
            agent.main_agent.save(champion_main_model_path)
        if not os.path.exists(champion_placement_model_path):
            print(
                "Champion placement model missing; bootstrapping from challenger current weights."
            )
            os.makedirs(os.path.dirname(champion_placement_model_path), exist_ok=True)
            agent.placement_agent.save(champion_placement_model_path)
        env = make_env(
            enemy_type="rl-capstone",
            enemy_ab_depth=args.enemy_ab_depth,
            enemy_mcts_n=args.enemy_mcts_n,
            enemy_greedy_n=args.enemy_greedy_n,
            enemy_ab_prunning=args.enemy_ab_prunning,
            map_template=resolved_map_template,
            map_mode=args.map_mode,
            fixed_map_seed=args.fixed_map_seed,
            enemy_main_model_path=champion_main_model_path,
            enemy_placement_model_path=champion_placement_model_path,
        )
    main_params = sum(p.numel() for p in agent.main_agent.model.parameters())
    pa = agent.placement_agent
    if hasattr(pa, "model"):
        place_desc = f"{sum(p.numel() for p in pa.model.parameters()):,} params"
    elif args.placement_strategy == "alphabeta":
        place_desc = (
            f"alphabeta(depth={args.placement_ab_depth}, "
            f"prunning={args.placement_ab_prunning})"
        )
    else:
        place_desc = args.placement_strategy
    device = get_device()

    print(
        f"Main agent ready      ({main_params:,} params, obs={FEATURE_SPACE_SIZE}, actions=245)\n"
        f"Placement agent ready ({place_desc})\n"
        f"Device: {device}"
    )

    effective_enemy = (
        first_phase.enemy_type
        if first_phase
        else (first_mix_enemy.enemy_type if first_mix_enemy else args.enemy)
    )
    effective_ab_depth = (
        _resolved_enemy_param(first_phase.enemy_type, first_phase.enemy_ab_depth, args)
        if first_phase
        else (
            _resolved_enemy_param(first_mix_enemy.enemy_type, first_mix_enemy.enemy_ab_depth, args)
            if first_mix_enemy
            else _resolved_enemy_param(args.enemy, None, args)
        )
    )
    enemy_detail = effective_enemy
    if effective_enemy in {"alphabeta", "alphabeta-prune", "same-turn-ab"}:
        enemy_detail += f" (depth={effective_ab_depth}"
        if effective_enemy == "alphabeta":
            enemy_detail += f", prunning={args.enemy_ab_prunning}"
        enemy_detail += ")"
    elif effective_enemy == "mcts":
        enemy_detail += f" (n={effective_ab_depth})"
    elif effective_enemy == "greedy":
        enemy_detail += f" (n={effective_ab_depth})"
    if schedule_phases:
        enemy_detail += " [fixed schedule enabled]"
    if args.enemy_smooth_mix:
        start_desc = ", ".join(
            f"{_enemy_spec_label(k, args.enemy_ab_depth)}:{v:.2f}"
            for k, v in sorted(start_mix.items(), key=lambda x: (x[0].enemy_type, x[0].enemy_ab_depth or -1))
        )
        end_desc = ", ".join(
            f"{_enemy_spec_label(k, args.enemy_ab_depth)}:{v:.2f}"
            for k, v in sorted(end_mix.items(), key=lambda x: (x[0].enemy_type, x[0].enemy_ab_depth or -1))
        )
        enemy_detail += (
            f" [smooth-mix seed={args.enemy_mix_seed}, "
            f"blend_games={args.enemy_mix_games}, start=({start_desc}), end=({end_desc})]"
        )
    if args.self_play_ladder:
        enemy_detail = (
            "rl-capstone [frozen champion: "
            f"main={champion_main_model_path}, placement={champion_placement_model_path}]"
        )
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
        if args.save and args.save_every_games > 0:
            training_detail += f", periodic_save_every={args.save_every_games} games"
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
        f"  Self-play ladder: {'enabled' if args.self_play_ladder else 'disabled'}"
        + (
            f"\n  Champion history dir: {args.champion_history_dir}"
            if args.self_play_ladder and args.champion_history_dir
            else ""
        )
        + f"\n  Map: {map_detail}\n"
        + f"  Training: {training_detail}\n"
        + f"  Replay JSON save: {replay_detail}\n"
        + f"  Benchmark CSV: {'disabled' if args.no_benchmark else args.benchmark_csv}\n"
        + f"  Device: {device}"
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

    recent_wins_by_enemy: Dict[str, deque] = {}
    promotions = 0
    games_completed = 0
    interrupted = False
    placement_save = (
        args.save_placement_model
        if args.placement_strategy == "model"
        else None
    )

    current_schedule_phase_index = 0 if schedule_phases else None
    current_mix_enemy: Optional[EnemySpec] = None if args.enemy_smooth_mix else None
    last_switch_log_game = -10**9

    try:
        for g in range(1, args.games + 1):
            active_enemy_for_game = _enemy_spec_label(
                EnemySpec(args.enemy, _resolved_enemy_param(args.enemy, None, args)),
                args.enemy_ab_depth,
            )
            if schedule_phases:
                next_phase_index, next_phase = schedule_phase_for_game(g, schedule_phases)
                active_enemy_for_game = _enemy_spec_label(
                    EnemySpec(
                        next_phase.enemy_type,
                        _resolved_enemy_param(
                            next_phase.enemy_type, next_phase.enemy_ab_depth, args
                        ),
                    ),
                    args.enemy_ab_depth,
                )
                if (
                    current_schedule_phase_index is None
                    or next_phase_index != current_schedule_phase_index
                ):
                    phase_depth = _resolved_enemy_param(
                        next_phase.enemy_type, next_phase.enemy_ab_depth, args
                    )
                    env = make_env(
                        enemy_type=next_phase.enemy_type,
                        enemy_ab_depth=phase_depth,
                        enemy_mcts_n=(
                            phase_depth if next_phase.enemy_type == "mcts" else args.enemy_mcts_n
                        ),
                        enemy_greedy_n=(
                            phase_depth if next_phase.enemy_type == "greedy" else args.enemy_greedy_n
                        ),
                        enemy_ab_prunning=args.enemy_ab_prunning,
                        map_template=resolved_map_template,
                        map_mode=args.map_mode,
                        fixed_map_seed=args.fixed_map_seed,
                    )
                    current_schedule_phase_index = next_phase_index
                    if (
                        args.enemy_switch_log_every > 0
                        and g - last_switch_log_game >= args.enemy_switch_log_every
                    ):
                        print(
                            "  switched scheduled enemy phase: "
                            f"phase={next_phase_index + 1}/{len(schedule_phases)} "
                            f"enemy={next_phase.enemy_type} "
                            f"games={next_phase.games} "
                            f"param={phase_depth}"
                        )
                        last_switch_log_game = g
            if args.enemy_smooth_mix:
                active_mix = interpolate_enemy_mix(
                    start_mix=start_mix,
                    end_mix=end_mix,
                    game_index_1_based=g,
                    blend_games=args.enemy_mix_games,
                )
                sampled_enemy = sample_enemy_from_mix(active_mix, mix_rng)
                active_enemy_for_game = _enemy_spec_label(
                    EnemySpec(
                        sampled_enemy.enemy_type,
                        _resolved_enemy_param(
                            sampled_enemy.enemy_type, sampled_enemy.enemy_ab_depth, args
                        ),
                    ),
                    args.enemy_ab_depth,
                )
                if current_mix_enemy != sampled_enemy:
                    phase_depth = _resolved_enemy_param(
                        sampled_enemy.enemy_type, sampled_enemy.enemy_ab_depth, args
                    )
                    env = make_env(
                        enemy_type=sampled_enemy.enemy_type,
                        enemy_ab_depth=phase_depth,
                        enemy_mcts_n=(
                            phase_depth if sampled_enemy.enemy_type == "mcts" else args.enemy_mcts_n
                        ),
                        enemy_greedy_n=(
                            phase_depth if sampled_enemy.enemy_type == "greedy" else args.enemy_greedy_n
                        ),
                        enemy_ab_prunning=args.enemy_ab_prunning,
                        map_template=resolved_map_template,
                        map_mode=args.map_mode,
                        fixed_map_seed=args.fixed_map_seed,
                    )
                    current_mix_enemy = sampled_enemy
                    if (
                        args.enemy_switch_log_every > 0
                        and g - last_switch_log_game >= args.enemy_switch_log_every
                    ):
                        mix_desc = ", ".join(
                            f"{_enemy_spec_label(k, args.enemy_ab_depth)}:{v:.2f}"
                            for k, v in sorted(
                                active_mix.items(),
                                key=lambda x: (x[0].enemy_type, x[0].enemy_ab_depth or -1),
                            )
                        )
                        print(
                            "  switched smooth-mix enemy: "
                            f"enemy={_enemy_spec_label(EnemySpec(sampled_enemy.enemy_type, phase_depth), args.enemy_ab_depth)} "
                            f"(mix={mix_desc})"
                        )
                        last_switch_log_game = g

            if args.self_play_ladder:
                active_enemy_for_game = "rl-capstone(champion)"

            keep_game_for_training = True
            if args.train:
                buffer_snapshot = (
                    snapshot_agent_buffer_lengths(agent)
                    if args.self_play_ladder and args.self_play_winner_only
                    else None
                )
                result = simulate_game(
                    agent, env, verbose=args.verbose, store_in_buffer=True
                )
                if args.self_play_ladder and args.self_play_winner_only and not result.won:
                    keep_game_for_training = False
                    truncate_agent_buffers_to_snapshot(agent, buffer_snapshot or {})
                if keep_game_for_training:
                    games_since_update += 1
                    last_bootstrap_obs = result.bootstrap_obs
                    last_bootstrap_mask = result.bootstrap_mask
            else:
                result = simulate_game(agent, env, verbose=args.verbose)

            wins += result.won
            losses += result.terminated and not result.won
            truncations += result.truncated
            recent_wins.append(int(result.won))
            enemy_recent = recent_wins_by_enemy.setdefault(
                active_enemy_for_game, deque(maxlen=300)
            )
            enemy_recent.append(int(result.won))
            rolling_n = len(recent_wins)
            rolling_300_win_rate = sum(recent_wins) / rolling_n if rolling_n > 0 else 0.0
            per_enemy_parts = []
            for enemy_key in sorted(recent_wins_by_enemy.keys()):
                samples = recent_wins_by_enemy[enemy_key]
                if len(samples) < args.per_enemy_min_samples:
                    continue
                wr = sum(samples) / len(samples)
                per_enemy_parts.append(
                    f"{enemy_key}={wr:.1%}({len(samples)})"
                )
            per_enemy_summary = (
                ", ".join(per_enemy_parts)
                if per_enemy_parts
                else f"- (min_samples={args.per_enemy_min_samples})"
            )

            status = "WON" if result.won else ("TRUNCATED" if result.truncated else "LOST")
            print(
                f"Game {g:4d}/{args.games}:  {status:>9s}  "
                f"steps={result.steps:4d}  reward={result.cumulative_reward:+.1f}  "
                f"[wr300={rolling_300_win_rate:.1%}({rolling_n}); by_enemy: {per_enemy_summary}]"
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

                if args.save and args.save_every_games > 0 and g % args.save_every_games == 0:
                    _ensure_parent_dir(args.save)
                    _ensure_parent_dir(placement_save)
                    agent.save(args.save, placement_save)
                    print(
                        f"  periodic checkpoint saved at game {g}: "
                        f"main={args.save}"
                    )

            if (
                args.self_play_ladder
                and args.train
                and g % args.self_play_eval_every_games == 0
            ):
                # Save challenger snapshot for evaluation and potential promotion.
                if args.save:
                    _ensure_parent_dir(args.save)
                    challenger_main_eval_path = args.save
                else:
                    challenger_main_eval_path = DEFAULT_MAIN_PLAY_MODEL_PATH
                if args.save_placement_model:
                    _ensure_parent_dir(args.save_placement_model)
                    challenger_placement_eval_path = args.save_placement_model
                else:
                    challenger_placement_eval_path = DEFAULT_PLACEMENT_MODEL_PATH
                agent.save(challenger_main_eval_path, challenger_placement_eval_path)

                eval_result = evaluate_challenger_vs_champion(
                    num_games=args.self_play_eval_games,
                    challenger_main_model_path=challenger_main_eval_path,
                    challenger_placement_model_path=challenger_placement_eval_path,
                    champion_main_model_path=champion_main_model_path,
                    champion_placement_model_path=champion_placement_model_path,
                    map_template=resolved_map_template,
                    map_mode=args.map_mode,
                    fixed_map_seed=args.fixed_map_seed,
                )
                eval_wr = eval_result["win_rate"]
                print(
                    "  self-play eval: "
                    f"challenger_wins={eval_result['wins']}/{eval_result['games']} "
                    f"win_rate={eval_wr:.1%} "
                    f"(promotion threshold={args.self_play_promotion_threshold:.1%})"
                )
                if eval_wr >= args.self_play_promotion_threshold:
                    shutil.copy2(challenger_main_eval_path, champion_main_model_path)
                    if challenger_placement_eval_path:
                        shutil.copy2(
                            challenger_placement_eval_path, champion_placement_model_path
                        )
                    promotions += 1
                    print(
                        f"  PROMOTION #{promotions}: challenger -> champion "
                        f"(new champion win_rate threshold met)"
                    )
                    if args.champion_history_dir:
                        os.makedirs(args.champion_history_dir, exist_ok=True)
                        hist_main, hist_place = _champion_history_paths(
                            args.champion_history_dir,
                            promotions,
                            g,
                            eval_wr,
                        )
                        shutil.copy2(champion_main_model_path, hist_main)
                        if (
                            challenger_placement_eval_path
                            and os.path.isfile(champion_placement_model_path)
                        ):
                            shutil.copy2(champion_placement_model_path, hist_place)
                            print(
                                f"  champion history saved:\n"
                                f"    main      {hist_main}\n"
                                f"    placement {hist_place}"
                            )
                        else:
                            print(f"  champion history saved:\n    main {hist_main}")
                    env = make_env(
                        enemy_type="rl-capstone",
                        enemy_ab_depth=args.enemy_ab_depth,
                        enemy_mcts_n=args.enemy_mcts_n,
                        enemy_greedy_n=args.enemy_greedy_n,
                        enemy_ab_prunning=args.enemy_ab_prunning,
                        map_template=resolved_map_template,
                        map_mode=args.map_mode,
                        fixed_map_seed=args.fixed_map_seed,
                        enemy_main_model_path=champion_main_model_path,
                        enemy_placement_model_path=champion_placement_model_path,
                    )
            games_completed = g
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user; flushing pending updates and saving latest weights...")

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
    if interrupted:
        print(
            f"Results: {wins}W / {losses}L / {truncations}T  "
            f"(completed {games_completed}/{args.games} games)"
        )
    else:
        print(f"Results: {wins}W / {losses}L / {truncations}T  ({args.games} games)")
    if args.self_play_ladder:
        print(
            f"Self-play ladder promotions: {promotions} "
            f"(champion main={champion_main_model_path})"
        )
    if benchmark is not None:
        print(f"Benchmarks logged to: {args.benchmark_csv} (run_name={run_name})")

    if args.train and args.save:
        main_ckpt = _timestamped_path(args.save)
        place_ckpt = _timestamped_path(placement_save) if placement_save else None

        _ensure_parent_dir(args.save)
        _ensure_parent_dir(placement_save)
        agent.save(args.save, placement_save)
        agent.save(main_ckpt, place_ckpt)

        print(f"Main model saved to {args.save}  (checkpoint: {main_ckpt})")
        if placement_save:
            print(f"Placement model saved to {placement_save}  (checkpoint: {place_ckpt})")


if __name__ == "__main__":
    main()