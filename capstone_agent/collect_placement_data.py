"""Collect supervised-learning data for the PlacementAgent.

Runs full games between two catanatron bots, records every placement
decision made by the Blue player (obs, mask, action from Blue's
perspective), and labels each with whether Blue won.

Both players use real strategies, so the placement data reflects
genuinely good (or at least intentional) decisions -- not random noise.

Games run in parallel across CPU cores for faster collection.
A background monitor prints per-worker progress every 60 seconds.
Supports graceful Ctrl+C (saves data collected so far) and --append
to resume collection into an existing .npz file.

Output: a .npz file containing:
    obs     (N, 1259)  float32  -- board observation at decision time
    masks   (N, 245)   float32  -- valid-action mask
    actions (N,)       int64    -- capstone action index chosen
    won     (N,)       float32  -- 1.0 if Blue won, 0.0 otherwise

Usage:
    python capstone_agent/collect_placement_data.py --games 5000

    # Use 8 workers instead of all cores:
    python capstone_agent/collect_placement_data.py --games 5000 --workers 8

    # Resume into an existing file:
    python capstone_agent/collect_placement_data.py --games 5000 \
        --append capstone_agent/data/placement_data_20260319T1530Z.npz
"""

import sys
import os
import argparse
import time
import threading
import multiprocessing as mp
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.gym.envs.capstone_env import (
    ACTION_SPACE_SIZE,
    to_action_space as capstone_action_index,
)
from catanatron.gym.envs.action_translator import (
    catanatron_action_to_capstone_index,
)

# ---------------------------------------------------------------------------
# Shared state set by pool initializer (each worker process gets a copy of
# the references to the shared arrays).
# ---------------------------------------------------------------------------
_worker_games = None
_worker_wins = None
_num_workers = 1


def _init_worker(games_arr, wins_arr, num_workers):
    global _worker_games, _worker_wins, _num_workers
    _worker_games = games_arr
    _worker_wins = wins_arr
    _num_workers = num_workers


def _make_bot(type_str, color):
    """Create a catanatron bot by name."""
    if type_str == "random":
        return RandomPlayer(color)
    elif type_str == "weighted":
        return WeightedRandomPlayer(color)
    elif type_str == "vp":
        return VictoryPointPlayer(color)
    elif type_str == "alphabeta":
        return AlphaBetaPlayer(color)
    elif type_str == "alphabeta-prune":
        return AlphaBetaPlayer(color, prunning=True)
    elif type_str == "same-turn-ab":
        return SameTurnAlphaBetaPlayer(color)
    elif type_str == "value":
        return ValueFunctionPlayer(color)
    else:
        raise ValueError(f"Unknown bot type: {type_str!r}")


BOT_CHOICES = sorted([
    "alphabeta", "alphabeta-prune", "random", "same-turn-ab",
    "value", "vp", "weighted",
])


def _make_action_mask(game):
    """Build a 245-dim binary mask of valid capstone actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for a in game.playable_actions:
        mask[capstone_action_index(a)] = 1.0
    return mask


def _play_one_game(args):
    """Play a single game and return placement data. Runs in a worker."""
    blue_type, enemy_type = args
    self_color = Color.BLUE
    opp_color = Color.RED

    blue = _make_bot(blue_type, self_color)
    red = _make_bot(enemy_type, opp_color)
    game = Game(players=[blue, red])

    obs_list, mask_list, action_list = [], [], []

    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
        current_color = game.state.current_color()
        is_placement = game.state.is_initial_build_phase

        if is_placement and current_color == self_color:
            obs = np.array(
                get_capstone_observation(game, self_color, opp_color),
                dtype=np.float32,
            )
            mask = _make_action_mask(game)
            obs_list.append(obs)
            mask_list.append(mask)

        action_record = game.play_tick()

        if is_placement and current_color == self_color:
            cap_idx = catanatron_action_to_capstone_index(action_record.action)
            action_list.append(cap_idx)

    won = float(game.winning_color() == self_color)

    slot = (mp.current_process()._identity[0] - 1) % _num_workers
    _worker_games[slot] += 1
    if won > 0.5:
        _worker_wins[slot] += 1

    return obs_list, mask_list, action_list, won


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _worker_bar(counts, width=20):
    """Tiny ASCII bar showing relative per-worker progress."""
    mx = max(counts) if max(counts) > 0 else 1
    return "".join("█" if c >= mx * 0.9 else "▓" if c >= mx * 0.5 else "░"
                    for c in counts)


def _monitor(worker_games, worker_wins, num_workers, total_games,
             stop_event, t0, state):
    """Background thread: prints a rich status line every 60 seconds."""
    while not stop_event.wait(60):
        elapsed = time.time() - t0
        counts = [worker_games[i] for i in range(num_workers)]
        wins = [worker_wins[i] for i in range(num_workers)]
        done = sum(counts)
        total_wins = sum(wins)
        samples = state["samples"]

        if done == 0:
            print(
                f"  [{_fmt_time(elapsed)}] Waiting for first results ...",
                flush=True,
            )
            continue

        rate = done / (elapsed / 60)
        eta_s = (total_games - done) / (done / elapsed) if done else 0
        win_pct = total_wins / done
        pct = done / total_games * 100

        # Line 1: aggregate stats
        line1 = (
            f"  [{_fmt_time(elapsed)}] "
            f"{done:>{len(str(total_games))}}/{total_games} ({pct:.0f}%)  "
            f"{samples} samples  "
            f"win={win_pct:.0%}  "
            f"{rate:.1f} games/min  "
            f"~{_fmt_time(eta_s)} left"
        )

        # Line 2: per-worker breakdown
        if num_workers <= 12:
            wk = "  ".join(
                f"W{i + 1}:{counts[i]}" for i in range(num_workers)
            )
        else:
            lo, hi = min(counts), max(counts)
            wk = (
                f"{num_workers} workers: {lo}–{hi} games each  "
                f"|{_worker_bar(counts)}|"
            )

        print(f"{line1}\n          {wk}", flush=True)


# ---------------------------------------------------------------------------
# Save / collect
# ---------------------------------------------------------------------------

def _save(out_path, all_obs, all_masks, all_actions, all_won):
    """Bundle lists into arrays and write a compressed .npz.

    Writes to a temporary file first, then atomically renames so the
    target file is never left in a half-written state.
    """
    if len(all_obs) == 0:
        print("  No data to save.")
        return

    obs = np.array(all_obs, dtype=np.float32)
    masks = np.array(all_masks, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)
    won = np.array(all_won, dtype=np.float32)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(out_path)
    tmp_path = base + ".tmp" + ext
    np.savez_compressed(tmp_path, obs=obs, masks=masks, actions=actions, won=won)
    os.replace(tmp_path, out_path)

    n_wins = int((won > 0.5).sum())
    print(
        f"  Saved {len(obs)} samples to {out_path}  "
        f"({n_wins} from wins, {len(obs) - n_wins} from losses)",
        flush=True,
    )


def collect(
    num_games: int,
    blue_type: str = "alphabeta",
    enemy_type: str = "alphabeta",
    out_path: str = "capstone_agent/data/placement_data.npz",
    append_path: str = None,
    num_workers: int = None,
):
    from action_map import validate as validate_action_mapping
    validate_action_mapping()

    for name, bot_type in [("blue", blue_type), ("enemy", enemy_type)]:
        if bot_type not in BOT_CHOICES:
            raise ValueError(
                f"Unknown {name} type {bot_type!r}. "
                f"Choose from: {', '.join(BOT_CHOICES)}"
            )

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    if append_path and os.path.exists(append_path):
        existing = np.load(append_path)
        all_obs = list(existing["obs"])
        all_masks = list(existing["masks"])
        all_actions = list(existing["actions"])
        all_won = list(existing["won"])
        print(f"  Loaded {len(all_obs)} existing samples from {append_path}")
    else:
        all_obs, all_masks, all_actions, all_won = [], [], [], []

    print(f"  Using {num_workers} worker processes")

    # Shared per-worker counters readable from the monitor thread
    worker_games = mp.Array("i", num_workers)
    worker_wins = mp.Array("i", num_workers)

    # Dict readable by monitor thread (same process → GIL-safe)
    state = {"samples": len(all_obs)}

    t0 = time.time()
    last_save_count = len(all_obs)
    games_done = 0
    autosave_every = max(50, num_games // 10)
    interrupted = False

    stop_monitor = threading.Event()
    mon = threading.Thread(
        target=_monitor,
        args=(worker_games, worker_wins, num_workers, num_games,
              stop_monitor, t0, state),
        daemon=True,
    )
    mon.start()

    work_items = [(blue_type, enemy_type)] * num_games

    try:
        with mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(worker_games, worker_wins, num_workers),
        ) as pool:
            for result in pool.imap_unordered(
                _play_one_game, work_items, chunksize=4
            ):
                obs_list, mask_list, action_list, won = result
                for o, m, a in zip(obs_list, mask_list, action_list):
                    all_obs.append(o)
                    all_masks.append(m)
                    all_actions.append(a)
                    all_won.append(won)

                games_done += 1
                state["samples"] = len(all_obs)

                if (
                    games_done % autosave_every == 0
                    and len(all_obs) > last_save_count
                ):
                    _save(
                        out_path, all_obs, all_masks, all_actions, all_won
                    )
                    last_save_count = len(all_obs)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n  Interrupted after {games_done} games.", flush=True)
    finally:
        stop_monitor.set()
        mon.join(timeout=2)

    _save(out_path, all_obs, all_masks, all_actions, all_won)

    # Final summary
    elapsed = time.time() - t0
    per_worker = [worker_games[i] for i in range(num_workers)]
    rate = games_done / (elapsed / 60) if elapsed > 0 else 0
    print(
        f"\n  Done: {games_done} games, {len(all_obs)} samples "
        f"in {_fmt_time(elapsed)}  ({rate:.1f} games/min)"
    )
    if num_workers <= 12:
        wk = "  ".join(
            f"W{i + 1}:{per_worker[i]}" for i in range(num_workers)
        )
    else:
        wk = (
            f"{min(per_worker)}–{max(per_worker)} games each  "
            f"|{_worker_bar(per_worker)}|"
        )
    print(f"  Per-worker: {wk}", flush=True)

    if interrupted:
        print(f"  Resume with: --append {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect placement training data from bot-vs-bot games"
    )
    parser.add_argument(
        "--games", type=int, default=5000, help="Number of games to play"
    )
    parser.add_argument(
        "--blue",
        type=str,
        default="alphabeta",
        choices=BOT_CHOICES,
        help="Bot type for Blue (the player we learn from)",
    )
    parser.add_argument(
        "--enemy",
        type=str,
        default="alphabeta",
        choices=BOT_CHOICES,
        help="Bot type for Red (the opponent)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="capstone_agent/data/placement_data.npz",
        help="Output .npz path (timestamp appended automatically)",
    )
    parser.add_argument(
        "--append",
        type=str,
        default=None,
        help="Path to existing .npz to resume from (new data is appended)",
    )
    args = parser.parse_args()

    if args.append:
        out_path = args.append
    else:
        root, ext = os.path.splitext(args.out)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        out_path = f"{root}_{stamp}{ext}"

    print(
        f"Collecting placement data from {args.games} games "
        f"(blue={args.blue} vs enemy={args.enemy}) ..."
    )
    if args.append:
        print(f"Appending to {args.append}")

    collect(
        args.games,
        blue_type=args.blue,
        enemy_type=args.enemy,
        out_path=out_path,
        append_path=args.append,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
