"""Interleaved compact supervised-learning loop for placement."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PlacementAgent import make_placement_agent
from benchmark_placement import HybridPlayer, run_group
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from CONSTANTS import PLACEMENT_AGENT_HIDDEN_SIZE
from train_compact_placement_supervised import train as train_compact_supervised

STATE_VERSION = 1


@dataclass(frozen=True)
class ChunkRecord:
    chunk_file: str
    chunk_path: str
    games_saved: int
    winner_only_examples: int
    manifest_path: str


@dataclass(frozen=True)
class TrainingSnapshot:
    new_chunk_files: list[str]
    replay_chunk_files: list[str]
    selected_chunk_files: list[str]
    new_games: int
    replay_games: int
    new_examples: int
    replay_examples: int


@dataclass(frozen=True)
class OutputPaths:
    run_dir: str
    data_dir: str
    models_dir: str
    logs_dir: str
    state_path: str
    status_log_path: str
    training_metrics_log_path: str
    latest_model_path: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")


def append_jsonl(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def atomic_write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def create_output_paths(run_dir: str) -> OutputPaths:
    run_dir = os.path.abspath(run_dir)
    data_dir = os.path.join(run_dir, "data")
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    return OutputPaths(
        run_dir=run_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        state_path=os.path.join(run_dir, "state.json"),
        status_log_path=os.path.join(logs_dir, "status.jsonl"),
        training_metrics_log_path=os.path.join(logs_dir, "training_metrics.jsonl"),
        latest_model_path=os.path.join(models_dir, "placement_model_latest.pt"),
    )


def default_state(paths: OutputPaths, base_seed: int) -> dict:
    return {
        "state_version": STATE_VERSION,
        "run_dir": paths.run_dir,
        "data_dir": paths.data_dir,
        "models_dir": paths.models_dir,
        "logs_dir": paths.logs_dir,
        "latest_model_path": None,
        "latest_checkpoint_path": None,
        "latest_benchmark": None,
        "next_collection_seed": int(base_seed),
        "collection_windows_started": 0,
        "collection_windows_completed": 0,
        "train_cycles_completed": 0,
        "trained_new_chunk_files": [],
    }


def load_or_initialize_state(paths: OutputPaths, base_seed: int) -> dict:
    if not os.path.exists(paths.state_path):
        return default_state(paths, base_seed)

    with open(paths.state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    if state.get("state_version") != STATE_VERSION:
        raise ValueError(
            f"Unsupported interleaved coordinator state version: {state.get('state_version')}"
        )

    state.setdefault("latest_model_path", None)
    state.setdefault("latest_checkpoint_path", None)
    state.setdefault("latest_benchmark", None)
    state.setdefault("next_collection_seed", int(base_seed))
    state.setdefault("collection_windows_started", 0)
    state.setdefault("collection_windows_completed", 0)
    state.setdefault("train_cycles_completed", 0)
    state.setdefault("trained_new_chunk_files", [])

    state["run_dir"] = paths.run_dir
    state["data_dir"] = paths.data_dir
    state["models_dir"] = paths.models_dir
    state["logs_dir"] = paths.logs_dir
    return state


def load_completed_chunk_records(data_dir: str) -> list[ChunkRecord]:
    manifest_paths = sorted(glob.glob(os.path.join(data_dir, "*.manifest.jsonl")))
    records_by_file: dict[str, ChunkRecord] = {}

    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") != "chunk_saved":
                    continue
                chunk_file = row["chunk_file"]
                chunk_path = os.path.join(data_dir, chunk_file)
                if not os.path.exists(chunk_path):
                    continue
                records_by_file.setdefault(
                    chunk_file,
                    ChunkRecord(
                        chunk_file=chunk_file,
                        chunk_path=chunk_path,
                        games_saved=int(row.get("games_saved", 0)),
                        winner_only_examples=int(row.get("winner_only_examples", 0)),
                        manifest_path=manifest_path,
                    ),
                )

    return sorted(records_by_file.values(), key=lambda record: record.chunk_file)


def new_chunk_records(
    chunk_records: list[ChunkRecord],
    trained_new_chunk_files: set[str],
) -> list[ChunkRecord]:
    return [
        record for record in chunk_records if record.chunk_file not in trained_new_chunk_files
    ]


def should_trigger_training(
    chunk_records: list[ChunkRecord],
    trained_new_chunk_files: set[str],
    min_new_chunks: int,
    min_new_games: int | None,
) -> bool:
    pending = new_chunk_records(chunk_records, trained_new_chunk_files)
    if len(pending) < min_new_chunks:
        return False
    if min_new_games is None:
        return True
    return sum(record.games_saved for record in pending) >= min_new_games


def select_training_snapshot(
    chunk_records: list[ChunkRecord],
    trained_new_chunk_files: set[str],
    replay_chunk_ratio: float,
    replay_chunk_cap: int | None,
    rng: np.random.Generator,
) -> TrainingSnapshot:
    new_records = new_chunk_records(chunk_records, trained_new_chunk_files)
    old_records = [
        record for record in chunk_records if record.chunk_file in trained_new_chunk_files
    ]

    replay_target = int(np.ceil(len(new_records) * replay_chunk_ratio))
    if replay_chunk_cap is not None:
        replay_target = min(replay_target, replay_chunk_cap)
    replay_target = min(replay_target, len(old_records))

    replay_records: list[ChunkRecord] = []
    if replay_target > 0:
        chosen_indices = rng.choice(
            len(old_records),
            size=replay_target,
            replace=False,
        )
        replay_records = [
            old_records[int(idx)] for idx in np.atleast_1d(chosen_indices)
        ]

    selected_records = sorted(
        new_records + replay_records,
        key=lambda record: record.chunk_file,
    )
    return TrainingSnapshot(
        new_chunk_files=[record.chunk_file for record in new_records],
        replay_chunk_files=[record.chunk_file for record in replay_records],
        selected_chunk_files=[record.chunk_file for record in selected_records],
        new_games=sum(record.games_saved for record in new_records),
        replay_games=sum(record.games_saved for record in replay_records),
        new_examples=sum(record.winner_only_examples for record in new_records),
        replay_examples=sum(record.winner_only_examples for record in replay_records),
    )


def should_run_benchmark(cycle_idx: int, benchmark_every_cycles: int) -> bool:
    if benchmark_every_cycles <= 0:
        return False
    return cycle_idx % benchmark_every_cycles == 0


def confidence_interval(win_rate: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    delta = 1.96 * (win_rate * (1 - win_rate) / n) ** 0.5
    return win_rate - delta, win_rate + delta


def summarize_group(
    wins: int,
    losses: int,
    draws: int,
    elapsed: float,
    n: int,
) -> dict:
    win_rate = wins / n if n else 0.0
    lo, hi = confidence_interval(win_rate, n)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": n,
        "elapsed_seconds": elapsed,
        "win_rate": win_rate,
        "ci_low": lo,
        "ci_high": hi,
    }


def benchmark_latest_model(
    placement_model_path: str,
    num_games: int,
    verbose: bool = False,
) -> dict:
    baseline_group = run_group(
        "baseline",
        lambda: AlphaBetaPlayer(Color.BLUE),
        num_games,
        verbose=verbose,
    )
    baseline = summarize_group(*baseline_group, n=num_games)

    placement_agent = make_placement_agent(
        "model",
        hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
    )
    placement_agent.load(placement_model_path)
    test_group = run_group(
        "model",
        lambda: HybridPlayer(Color.BLUE, placement_agent),
        num_games,
        verbose=verbose,
    )
    test = summarize_group(*test_group, n=num_games)

    return {
        "games": num_games,
        "placement_model_path": placement_model_path,
        "baseline": baseline,
        "test": test,
        "delta_win_rate": test["win_rate"] - baseline["win_rate"],
    }


class CollectorManager(threading.Thread):
    def __init__(
        self,
        *,
        repo_root: str,
        paths: OutputPaths,
        state: dict,
        state_lock: threading.Lock,
        stop_event: threading.Event,
        pause_event: threading.Event,
        status_log_path: str,
        games_per_window: int,
        games_per_chunk: int,
        workers: int | None,
        ab_depth: int,
        poll_interval_s: float,
    ):
        super().__init__(daemon=True)
        self.repo_root = repo_root
        self.paths = paths
        self.state = state
        self.state_lock = state_lock
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.status_log_path = status_log_path
        self.games_per_window = games_per_window
        self.games_per_chunk = games_per_chunk
        self.workers = workers
        self.ab_depth = ab_depth
        self.poll_interval_s = poll_interval_s
        self.failure_message: str | None = None
        self.current_process: subprocess.Popen | None = None

    def collector_command(self, seed: int) -> list[str]:
        command = [
            sys.executable,
            os.path.join(
                self.repo_root,
                "capstone_agent",
                "Placement",
                "collect_compact_placement_data.py",
            ),
            "--games",
            str(self.games_per_window),
            "--games-per-chunk",
            str(self.games_per_chunk),
            "--out-dir",
            self.paths.data_dir,
            "--seed",
            str(seed),
            "--ab-depth",
            str(self.ab_depth),
        ]
        if self.workers is not None:
            command.extend(["--workers", str(self.workers)])
        return command

    def stop(self, immediate: bool = False):
        self.stop_event.set()
        if (
            immediate
            and self.current_process is not None
            and self.current_process.poll() is None
        ):
            self.current_process.terminate()

    def run(self):
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(self.poll_interval_s)
                continue

            with self.state_lock:
                seed = int(self.state["next_collection_seed"])
                window_index = int(self.state["collection_windows_started"]) + 1
                self.state["collection_windows_started"] = window_index
                self.state["next_collection_seed"] = seed + self.games_per_window
                atomic_write_json(self.paths.state_path, self.state)

            append_jsonl(
                self.status_log_path,
                {
                    "event": "collection_window_started",
                    "window_index": window_index,
                    "seed": seed,
                    "games_per_window": self.games_per_window,
                    "games_per_chunk": self.games_per_chunk,
                    "workers": self.workers,
                    "ab_depth": self.ab_depth,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            self.current_process = subprocess.Popen(
                self.collector_command(seed),
                cwd=self.repo_root,
            )
            return_code = self.current_process.wait()
            self.current_process = None

            append_jsonl(
                self.status_log_path,
                {
                    "event": "collection_window_finished",
                    "window_index": window_index,
                    "seed": seed,
                    "return_code": return_code,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            with self.state_lock:
                if return_code == 0:
                    self.state["collection_windows_completed"] = int(
                        self.state["collection_windows_completed"]
                    ) + 1
                atomic_write_json(self.paths.state_path, self.state)

            if return_code != 0:
                self.failure_message = (
                    f"collector window {window_index} exited with return code {return_code}"
                )
                self.stop_event.set()
                return


def initialize_run(args) -> tuple[OutputPaths, dict]:
    if args.run_dir is None:
        run_dir = os.path.join(
            "capstone_agent",
            "online_runs",
            f"compact_placement_{utc_timestamp()}",
        )
    else:
        run_dir = args.run_dir

    paths = create_output_paths(run_dir)
    for directory in (
        paths.run_dir,
        paths.data_dir,
        paths.models_dir,
        paths.logs_dir,
    ):
        os.makedirs(directory, exist_ok=True)

    state = load_or_initialize_state(paths, args.seed)
    atomic_write_json(paths.state_path, state)
    return paths, state


def print_waiting_status(
    pending_chunk_count: int,
    pending_games: int,
    pending_examples: int,
    target_chunks: int,
    target_games: int | None,
):
    game_target_str = "-" if target_games is None else str(target_games)
    print(
        "Waiting for more completed chunks before training  "
        f"new_chunks={pending_chunk_count}/{target_chunks}  "
        f"new_games={pending_games}/{game_target_str}  "
        f"new_examples={pending_examples}",
        flush=True,
    )


def print_benchmark_summary(result: dict):
    baseline = result["baseline"]
    test = result["test"]
    print(
        f"Benchmark  baseline={baseline['win_rate']:.1%} "
        f"[{baseline['ci_low']:.1%}, {baseline['ci_high']:.1%}]  "
        f"test={test['win_rate']:.1%} "
        f"[{test['ci_low']:.1%}, {test['ci_high']:.1%}]  "
        f"delta={result['delta_win_rate']:+.1%}",
        flush=True,
    )


def run(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paths, state = initialize_run(args)

    append_jsonl(
        paths.status_log_path,
        {
            "event": "run_started",
            "run_dir": paths.run_dir,
            "data_dir": paths.data_dir,
            "models_dir": paths.models_dir,
            "logs_dir": paths.logs_dir,
            "cycles": args.cycles,
            "collector_games_per_window": args.collector_games_per_window,
            "collector_games_per_chunk": args.collector_games_per_chunk,
            "collector_workers": args.collector_workers,
            "collector_ab_depth": args.collector_ab_depth,
            "min_new_chunks": args.min_new_chunks,
            "min_new_games": args.min_new_games,
            "replay_chunk_ratio": args.replay_chunk_ratio,
            "replay_chunk_cap": args.replay_chunk_cap,
            "benchmark_games": args.benchmark_games,
            "benchmark_every_cycles": args.benchmark_every_cycles,
            "selection_mode": args.selection_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    state_lock = threading.Lock()
    stop_event = threading.Event()
    pause_event = threading.Event()
    collector = CollectorManager(
        repo_root=repo_root,
        paths=paths,
        state=state,
        state_lock=state_lock,
        stop_event=stop_event,
        pause_event=pause_event,
        status_log_path=paths.status_log_path,
        games_per_window=args.collector_games_per_window,
        games_per_chunk=args.collector_games_per_chunk,
        workers=args.collector_workers,
        ab_depth=args.collector_ab_depth,
        poll_interval_s=args.poll_interval_s,
    )
    collector.start()

    replay_rng = np.random.default_rng(args.replay_seed)
    last_wait_print = 0.0

    print("Interleaved compact placement loop")
    print(f"  Run dir: {paths.run_dir}")
    print(f"  Data dir: {paths.data_dir}")
    print(f"  Latest model: {paths.latest_model_path}")
    print(f"  Status log: {paths.status_log_path}")
    collector_workers_str = str(args.collector_workers) if args.collector_workers is not None else "auto"
    print(f"  Collector workers: {collector_workers_str}")
    print()

    try:
        while True:
            if collector.failure_message is not None:
                raise RuntimeError(collector.failure_message)

            with state_lock:
                completed_cycles = int(state["train_cycles_completed"])
                trained_new = set(state["trained_new_chunk_files"])
                latest_model_path = state.get("latest_model_path")

            if completed_cycles >= args.cycles:
                break

            chunk_records = load_completed_chunk_records(paths.data_dir)
            pending_records = new_chunk_records(chunk_records, trained_new)
            pending_chunk_count = len(pending_records)
            pending_games = sum(record.games_saved for record in pending_records)
            pending_examples = sum(
                record.winner_only_examples for record in pending_records
            )

            if not should_trigger_training(
                chunk_records,
                trained_new,
                args.min_new_chunks,
                args.min_new_games,
            ):
                now = time.time()
                if now - last_wait_print >= max(30.0, args.poll_interval_s):
                    print_waiting_status(
                        pending_chunk_count,
                        pending_games,
                        pending_examples,
                        args.min_new_chunks,
                        args.min_new_games,
                    )
                    last_wait_print = now
                time.sleep(args.poll_interval_s)
                continue

            cycle_idx = completed_cycles + 1
            snapshot = select_training_snapshot(
                chunk_records,
                trained_new,
                args.replay_chunk_ratio,
                args.replay_chunk_cap,
                replay_rng,
            )
            selected_chunk_paths = [
                os.path.join(paths.data_dir, chunk_file)
                for chunk_file in snapshot.selected_chunk_files
            ]

            print(
                f"\n=== Training cycle {cycle_idx}/{args.cycles} ===\n"
                f"  new_chunks={len(snapshot.new_chunk_files)}  "
                f"replay_chunks={len(snapshot.replay_chunk_files)}\n"
                f"  new_games={snapshot.new_games}  "
                f"replay_games={snapshot.replay_games}\n"
                f"  latest_model={latest_model_path or 'fresh start'}",
                flush=True,
            )

            train_result = train_compact_supervised(
                chunk_paths=selected_chunk_paths,
                out_path=paths.latest_model_path,
                resume_path=latest_model_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden_size=args.hidden_size,
                val_frac=args.val_frac,
                selection_mode=args.selection_mode,
                win_weight=args.win_weight,
                loss_weight=args.loss_weight,
                split_seed=args.split_seed + cycle_idx,
                metrics_path=paths.training_metrics_log_path,
            )

            benchmark_result = None
            if should_run_benchmark(cycle_idx, args.benchmark_every_cycles):
                if args.pause_collection_during_benchmark:
                    pause_event.set()
                try:
                    benchmark_result = benchmark_latest_model(
                        paths.latest_model_path,
                        num_games=args.benchmark_games,
                        verbose=args.benchmark_verbose,
                    )
                finally:
                    pause_event.clear()
                print_benchmark_summary(benchmark_result)

            with state_lock:
                updated_trained = set(state["trained_new_chunk_files"])
                updated_trained.update(snapshot.new_chunk_files)
                state["trained_new_chunk_files"] = sorted(updated_trained)
                state["train_cycles_completed"] = cycle_idx
                state["latest_model_path"] = paths.latest_model_path
                state["latest_checkpoint_path"] = train_result["checkpoint_path"]
                if benchmark_result is not None:
                    state["latest_benchmark"] = benchmark_result
                atomic_write_json(paths.state_path, state)

            append_jsonl(
                paths.status_log_path,
                {
                    "event": "training_cycle_finished",
                    "cycle": cycle_idx,
                    "new_chunk_files": snapshot.new_chunk_files,
                    "replay_chunk_files": snapshot.replay_chunk_files,
                    "selected_chunk_files": snapshot.selected_chunk_files,
                    "new_games": snapshot.new_games,
                    "replay_games": snapshot.replay_games,
                    "new_examples": snapshot.new_examples,
                    "replay_examples": snapshot.replay_examples,
                    "train_result": train_result,
                    "benchmark_result": benchmark_result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    except KeyboardInterrupt:
        print("\nInterrupted interleaved training loop.", flush=True)
        append_jsonl(
            paths.status_log_path,
            {
                "event": "run_interrupted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    finally:
        collector.stop(immediate=False)
        collector.join()
        with state_lock:
            atomic_write_json(paths.state_path, state)

    append_jsonl(
        paths.status_log_path,
        {
            "event": "run_finished",
            "train_cycles_completed": state.get("train_cycles_completed", 0),
            "collection_windows_completed": state.get(
                "collection_windows_completed", 0
            ),
            "latest_model_path": state.get("latest_model_path"),
            "latest_checkpoint_path": state.get("latest_checkpoint_path"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("\nInterleaved compact placement loop finished.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Interleaved compact placement collect-train-benchmark loop"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory for data, models, logs, and state (default: timestamped run)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of training cycles to complete before stopping",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base collection seed for the first collector window",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=0,
        help="RNG seed for replay chunk sampling",
    )
    parser.add_argument(
        "--collector-games-per-window",
        type=int,
        default=1000,
        help="Games collected by each background collector window",
    )
    parser.add_argument(
        "--collector-games-per-chunk",
        type=int,
        default=1000,
        help="Attempted games per chunk written by each collector window",
    )
    parser.add_argument(
        "--collector-workers",
        "--workers",
        dest="collector_workers",
        type=int,
        default=None,
        help=(
            "Worker processes dedicated to background collection "
            "(alias: --workers)"
        ),
    )
    parser.add_argument(
        "--collector-ab-depth",
        type=int,
        default=2,
        help="AlphaBeta search depth for the collector's main-game policy",
    )
    parser.add_argument(
        "--min-new-chunks",
        type=int,
        default=1,
        help="Minimum number of unseen completed chunks required before training",
    )
    parser.add_argument(
        "--min-new-games",
        type=int,
        default=None,
        help="Optional minimum number of unseen saved games required before training",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=10.0,
        help="Seconds between coordinator polls while waiting for new chunks",
    )
    parser.add_argument(
        "--replay-chunk-ratio",
        type=float,
        default=1.0,
        help="Replay old chunks at this ratio relative to the number of new chunks",
    )
    parser.add_argument(
        "--replay-chunk-cap",
        type=int,
        default=None,
        help="Optional hard cap on replay chunk count per training cycle",
    )
    parser.add_argument(
        "--benchmark-games",
        type=int,
        default=100,
        help="Benchmark games after each evaluation cycle",
    )
    parser.add_argument(
        "--benchmark-every-cycles",
        type=int,
        default=1,
        help="Run the benchmark every N training cycles (0 disables)",
    )
    parser.add_argument(
        "--benchmark-verbose",
        action="store_true",
        help="Print per-50-game progress during benchmarks",
    )
    parser.add_argument(
        "--pause-collection-during-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause launching new collector windows during benchmark evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per cycle",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Training learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Training weight decay",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Placement model hidden size",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction for each training cycle",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="winner_only",
        choices=["winner_only", "outcome_weighted", "all_examples"],
        help="How to include winner/loser actions during training",
    )
    parser.add_argument(
        "--win-weight",
        type=float,
        default=1.0,
        help="Per-example weight for winner examples when outcome weighting is used",
    )
    parser.add_argument(
        "--loss-weight",
        type=float,
        default=0.1,
        help="Per-example weight for loser examples when outcome weighting is used",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Base split seed for train/validation partitioning",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
