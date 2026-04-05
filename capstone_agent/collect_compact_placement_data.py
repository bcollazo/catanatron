"""Collect chunked compact supervised data for opening placement."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from catanatron.game import Game
from catanatron.models.player import Color

from PlacementAgent import RandomPlacementAgent
from placement_supervised_dataset import (
    OPENING_STEP_COUNT,
    SCHEMA_VERSION,
    CompactPlacementAccumulator,
    save_chunk_records,
)
from router_search_player import AlphaBetaMainAgentAdapter, RouterCapstonePlayer


def _append_manifest_row(manifest_path: str, row: dict):
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _play_one_game(args):
    game_seed, ab_depth = args
    random.seed(game_seed)
    np.random.seed(game_seed % (2**32 - 1))

    blue = RouterCapstonePlayer(
        Color.BLUE,
        RandomPlacementAgent(),
        AlphaBetaMainAgentAdapter(Color.BLUE, depth=ab_depth),
    )
    red = RouterCapstonePlayer(
        Color.RED,
        RandomPlacementAgent(),
        AlphaBetaMainAgentAdapter(Color.RED, depth=ab_depth),
    )
    game = Game(players=[blue, red], seed=game_seed)
    accumulator = CompactPlacementAccumulator()
    winner = game.play(accumulators=[accumulator])

    if accumulator.record is None:
        return {
            "record": None,
            "skip_reason": accumulator.skip_reason or "unknown",
            "game_seed": game_seed,
        }

    return {
        "record": accumulator.record,
        "skip_reason": None,
        "winner_color": winner.value if winner is not None else None,
        "game_seed": game_seed,
    }


def _resolve_workers(requested_workers: int | None) -> int:
    if requested_workers is not None:
        return max(1, int(requested_workers))
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def _save_chunk(
    *,
    out_dir: str,
    run_prefix: str,
    chunk_index: int,
    attempted_in_chunk: int,
    records: list[dict],
    skipped_in_chunk: int,
    manifest_path: str,
):
    if len(records) == 0:
        return

    filename = f"{run_prefix}_chunk{chunk_index:04d}.npz"
    chunk_path = os.path.join(out_dir, filename)
    save_chunk_records(chunk_path, records)
    _append_manifest_row(
        manifest_path,
        {
            "event": "chunk_saved",
            "schema_version": SCHEMA_VERSION,
            "chunk_file": filename,
            "games_attempted": attempted_in_chunk,
            "games_saved": len(records),
            "games_skipped": skipped_in_chunk,
            "winner_only_examples": len(records) * (OPENING_STEP_COUNT // 2),
        },
    )


def collect(
    *,
    num_games: int,
    games_per_chunk: int,
    out_dir: str,
    num_workers: int | None = None,
    seed: int = 0,
    ab_depth: int = 2,
):
    os.makedirs(out_dir, exist_ok=True)
    num_workers = _resolve_workers(num_workers)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = f"placement_compact_v{SCHEMA_VERSION}_{timestamp}_seed{seed}"
    manifest_path = os.path.join(out_dir, f"{run_prefix}.manifest.jsonl")
    _append_manifest_row(
        manifest_path,
        {
            "event": "run_started",
            "schema_version": SCHEMA_VERSION,
            "num_games": num_games,
            "games_per_chunk": games_per_chunk,
            "workers": num_workers,
            "seed": seed,
            "ab_depth": ab_depth,
        },
    )

    seeds = [seed + game_idx for game_idx in range(num_games)]
    work_items = [(game_seed, ab_depth) for game_seed in seeds]

    attempted_total = 0
    saved_total = 0
    skipped_total = 0
    skip_reasons = Counter()

    attempted_in_chunk = 0
    skipped_in_chunk = 0
    pending_records: list[dict] = []
    chunk_index = 1

    t0 = time.time()
    last_report = t0
    interrupted = False

    pool = mp.Pool(processes=num_workers)
    try:
        for result in pool.imap_unordered(_play_one_game, work_items, chunksize=1):
            attempted_total += 1
            attempted_in_chunk += 1

            record = result["record"]
            if record is None:
                skipped_total += 1
                skipped_in_chunk += 1
                skip_reasons[result["skip_reason"]] += 1
            else:
                saved_total += 1
                pending_records.append(record)

            now = time.time()
            if now - last_report >= 60 or attempted_total == num_games:
                elapsed = now - t0
                rate = attempted_total / max(elapsed, 1e-6)
                eta = (num_games - attempted_total) / max(rate, 1e-6)
                print(
                    f"[{_format_elapsed(elapsed)}] "
                    f"{attempted_total}/{num_games} games  "
                    f"saved={saved_total}  skipped={skipped_total}  "
                    f"{rate:.2f} games/s  eta={_format_elapsed(eta)}",
                    flush=True,
                )
                last_report = now

            if attempted_in_chunk >= games_per_chunk:
                _save_chunk(
                    out_dir=out_dir,
                    run_prefix=run_prefix,
                    chunk_index=chunk_index,
                    attempted_in_chunk=attempted_in_chunk,
                    records=pending_records,
                    skipped_in_chunk=skipped_in_chunk,
                    manifest_path=manifest_path,
                )
                pending_records = []
                attempted_in_chunk = 0
                skipped_in_chunk = 0
                chunk_index += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving completed chunks and partial data...", flush=True)
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()

    if pending_records:
        _save_chunk(
            out_dir=out_dir,
            run_prefix=run_prefix,
            chunk_index=chunk_index,
            attempted_in_chunk=attempted_in_chunk,
            records=pending_records,
            skipped_in_chunk=skipped_in_chunk,
            manifest_path=manifest_path,
        )

    elapsed = time.time() - t0
    _append_manifest_row(
        manifest_path,
        {
            "event": "run_finished",
            "schema_version": SCHEMA_VERSION,
            "attempted_total": attempted_total,
            "saved_total": saved_total,
            "skipped_total": skipped_total,
            "skip_reasons": dict(skip_reasons),
            "elapsed_seconds": elapsed,
            "interrupted": interrupted,
        },
    )

    print(
        "\nFinished compact placement collection\n"
        f"  output_dir: {out_dir}\n"
        f"  manifest:   {manifest_path}\n"
        f"  attempted:  {attempted_total}\n"
        f"  saved:      {saved_total}\n"
        f"  skipped:    {skipped_total}\n"
        f"  elapsed:    {_format_elapsed(elapsed)}",
        flush=True,
    )
    if skip_reasons:
        print(f"  skip_reasons: {dict(skip_reasons)}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect chunked compact supervised data for opening placement"
    )
    parser.add_argument("--games", type=int, default=5000, help="Games to simulate")
    parser.add_argument(
        "--games-per-chunk",
        type=int,
        default=5000,
        help="Attempted games per saved chunk",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="capstone_agent/data/compact_placement",
        help="Directory for chunk files and manifest",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes to use (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; game i uses seed + i",
    )
    parser.add_argument(
        "--ab-depth",
        type=int,
        default=2,
        help="AlphaBeta search depth for the post-placement main-game policy",
    )
    args = parser.parse_args()

    collect(
        num_games=args.games,
        games_per_chunk=args.games_per_chunk,
        out_dir=args.out_dir,
        num_workers=args.workers,
        seed=args.seed,
        ab_depth=args.ab_depth,
    )


if __name__ == "__main__":
    main()
