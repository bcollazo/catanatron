import json
import os

import numpy as np

from capstone_agent.train_compact_placement_online import (
    ChunkRecord,
    atomic_write_json,
    create_output_paths,
    load_completed_chunk_records,
    load_or_initialize_state,
    select_training_snapshot,
    should_run_benchmark,
    should_trigger_training,
)


def test_load_completed_chunk_records_discovers_manifested_chunks(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    manifest_path = data_dir / "collector.manifest.jsonl"
    rows = [
        {
            "event": "chunk_saved",
            "chunk_file": "placement_b.npz",
            "games_saved": 3,
            "winner_only_examples": 12,
        },
        {
            "event": "chunk_saved",
            "chunk_file": "missing.npz",
            "games_saved": 10,
            "winner_only_examples": 40,
        },
        {
            "event": "chunk_saved",
            "chunk_file": "placement_a.npz",
            "games_saved": 5,
            "winner_only_examples": 20,
        },
    ]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    (data_dir / "placement_a.npz").write_bytes(b"a")
    (data_dir / "placement_b.npz").write_bytes(b"b")

    records = load_completed_chunk_records(str(data_dir))

    assert [record.chunk_file for record in records] == [
        "placement_a.npz",
        "placement_b.npz",
    ]
    assert [record.games_saved for record in records] == [5, 3]
    assert [record.winner_only_examples for record in records] == [20, 12]


def test_should_trigger_training_requires_all_configured_minimums():
    records = [
        ChunkRecord("old.npz", "/tmp/old.npz", games_saved=4, winner_only_examples=16, manifest_path="old"),
        ChunkRecord("new_a.npz", "/tmp/new_a.npz", games_saved=3, winner_only_examples=12, manifest_path="new"),
        ChunkRecord("new_b.npz", "/tmp/new_b.npz", games_saved=2, winner_only_examples=8, manifest_path="new"),
    ]
    trained_new = {"old.npz"}

    assert should_trigger_training(records, trained_new, min_new_chunks=2, min_new_games=5)
    assert not should_trigger_training(records, trained_new, min_new_chunks=3, min_new_games=5)
    assert not should_trigger_training(records, trained_new, min_new_chunks=2, min_new_games=6)
    assert should_trigger_training(records, trained_new, min_new_chunks=1, min_new_games=None)


def test_select_training_snapshot_uses_all_new_chunks_plus_bounded_replay():
    records = [
        ChunkRecord("old_a.npz", "/tmp/old_a.npz", games_saved=4, winner_only_examples=16, manifest_path="old"),
        ChunkRecord("old_b.npz", "/tmp/old_b.npz", games_saved=5, winner_only_examples=20, manifest_path="old"),
        ChunkRecord("new_a.npz", "/tmp/new_a.npz", games_saved=6, winner_only_examples=24, manifest_path="new"),
        ChunkRecord("new_b.npz", "/tmp/new_b.npz", games_saved=7, winner_only_examples=28, manifest_path="new"),
    ]
    trained_new = {"old_a.npz", "old_b.npz"}

    snapshot = select_training_snapshot(
        records,
        trained_new,
        replay_chunk_ratio=1.0,
        replay_chunk_cap=1,
        rng=np.random.default_rng(7),
    )

    assert snapshot.new_chunk_files == ["new_a.npz", "new_b.npz"]
    assert len(snapshot.replay_chunk_files) == 1
    assert set(snapshot.replay_chunk_files).issubset({"old_a.npz", "old_b.npz"})
    assert set(snapshot.selected_chunk_files) == set(snapshot.new_chunk_files) | set(snapshot.replay_chunk_files)
    assert snapshot.new_games == 13
    assert snapshot.replay_games in {4, 5}
    assert snapshot.new_examples == 52
    assert snapshot.replay_examples in {16, 20}


def test_state_round_trip_preserves_resume_state(tmp_path):
    paths = create_output_paths(str(tmp_path / "online_run"))
    for directory in (paths.run_dir, paths.data_dir, paths.models_dir, paths.logs_dir):
        os.makedirs(directory, exist_ok=True)

    state = load_or_initialize_state(paths, base_seed=11)
    state["next_collection_seed"] = 42
    state["train_cycles_completed"] = 3
    state["trained_new_chunk_files"] = ["chunk_a.npz", "chunk_b.npz"]
    atomic_write_json(paths.state_path, state)

    loaded = load_or_initialize_state(paths, base_seed=999)

    assert loaded["next_collection_seed"] == 42
    assert loaded["train_cycles_completed"] == 3
    assert loaded["trained_new_chunk_files"] == ["chunk_a.npz", "chunk_b.npz"]
    assert loaded["run_dir"] == paths.run_dir
    assert loaded["data_dir"] == paths.data_dir


def test_should_run_benchmark_uses_cycle_cadence():
    assert should_run_benchmark(1, 1)
    assert not should_run_benchmark(1, 2)
    assert should_run_benchmark(2, 2)
    assert not should_run_benchmark(3, 0)
