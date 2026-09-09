# Rust rollout architecture probes

Supporting experiments for [the implementation plan](../../RUST_IMPLEMENTATION_PLAN.md).
This directory is a dependency-free Rust benchmark program plus Python baseline/wire probes, **not a Catan engine**.

## Contents

* `baseline.py`: eight complete Python games, 128 real-board fixtures, copy/generation/full-game timings and profile.
* `fixtures.txt`: actual BASE edges followed by rows of player (1–4), 54 building owners and 72 road owners (0 means empty). Generated from base revision `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`, seeds 0–7, hash seed 0. Building type is irrelevant to these geometric queries.
* `src/main.rs`: equivalent array/bitset road queries; vector/bitset longest-road DFS with checks; enum/packed dispatch; fresh/reused/stack move buffers; direct selection; synthetic copy/undo probes.
* `protocol_probe.py`: PR #386's actual message builder and serializer, followed by a persistent JSON echo process. Does not benchmark the full host adapter or Rust parsing.
* `python-results.json`, `python-profile.txt`, `protocol-results.json`: raw baseline results.
* `rust-results-portable.csv`, `rust-results-native.csv`, `rust-results-native-repeat.csv`: medians, min/max and all nine timed batches in ns/op, plus checksums.

Recorded 2026-09-07 on Ryzen 9 9950X / Windows 11 build 26200; Python 3.12.14, NetworkX 3.5; rustc 1.90.0, LLVM 20.1.8, Windows MSVC target. No affinity or locked clocks. See the plan for interpretation and experimental limits. The entire fixture corpus is reused during a run; this does not simulate a large search working set. Copy/undo are byte-array surrogates, and action probes are synthetic workloads.

## Reproduce

From the repository root, use any Python >=3.11 with NetworkX 3.5. The baseline also recognizes a repo-local `.venv/planning-deps` directory, which was used for this run. It imports the checkout directly; no editable project install is needed.

PowerShell:

```powershell
python -m pip install networkx==3.5
$env:PYTHONHASHSEED = '0'
python experiments/rust-rollout/baseline.py
Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
cargo run --release --offline --manifest-path experiments/rust-rollout/Cargo.toml > experiments/rust-rollout/rust-results-portable.csv
$env:RUSTFLAGS = '-C target-cpu=native'
cargo run --release --offline --manifest-path experiments/rust-rollout/Cargo.toml > experiments/rust-rollout/rust-results-native.csv
& experiments/rust-rollout/target/release/catan-rollout-design-experiments.exe > experiments/rust-rollout/rust-results-native-repeat.csv
```

Use your normal virtual environment for the pip command. On this host `python` was an unconfigured pyenv shim, so the bundled CPython executable was used instead. Package installation and Python benchmarks needed expanded access because the sandbox could not read the newly installed package. Rust ran in the ordinary workspace sandbox. Downloads/build products live outside the committed results or in ignored `target/`.

On POSIX, use `PYTHONHASHSEED=0 python ...` and `RUSTFLAGS='-C target-cpu=native' cargo ...`; omit `.exe` when invoking the binary. Rust can use the committed fixtures without running Python or accessing the network. The harness runs assertions before printing measurements; an assertion failure must abort comparison. It verifies mask equality on all 128 fixtures and independent DFS agreement, with three additional expected-result graph cases. Baseline JSON records the Python road-query mismatch count (zero here).

For the PR wire probe, fetch exactly these two pinned source files into the ignored directory (GitHub CLI required for these example commands):

```powershell
New-Item -ItemType Directory -Force experiments/rust-rollout/target/pr386
gh api -H 'Accept: application/vnd.github.raw+json' 'repos/bcollazo/catanatron/contents/catanatron/catanatron/serialization.py?ref=5149b1869ba6318a2f2e3ef3925915576a433286' > experiments/rust-rollout/target/pr386/serialization.py
gh api -H 'Accept: application/vnd.github.raw+json' 'repos/bcollazo/catanatron/contents/catanatron/catanatron/protocol.py?ref=5149b1869ba6318a2f2e3ef3925915576a433286' > experiments/rust-rollout/target/pr386/protocol.py
$env:PYTHONHASHSEED = '0'
python experiments/rust-rollout/protocol_probe.py
```

During this run the files were read through the connected GitHub API tool because shell network access was restricted. The probe loads only these two modules from the pinned PR; it does not check out or modify that PR. Its serializer imports are compatible with the pinned base engine. Repeat on that revision for meaningful comparison.

## Measurement details

Rust uses `opt-level=3`, thin LTO, one codegen unit. Every row warms up 1,000 calls and runs nine timed batches. Iterations are included in CSV; checksums consume results and `black_box` inhibits elimination. Raw samples are semicolon-separated in the final CSV column. The loop-control measurement is reported, not subtracted. Very short operations remain sensitive to compiler optimization and scheduling.

Array/recomputed-mask road queries include local connectivity work; the cached-node query excludes cache maintenance. All return a mask, so move materialization is measured separately. Longest-road routines search edge-simple trails and count roads terminating at enemy buildings. They are independent Rust references, not a full oracle for Python's award logic. The synthetic enum is 11 bytes on the measured compiler; the packed workload is 8 bytes and encodes only four variants, not a public serialization format.

Both copy and undo expose the mutated full array to `black_box`, read the same selected byte, and produce matching checksums. Undo saves/restores eight known offsets and omits general undo-record construction, dynamic allocation and Catan cache restoration. It is intentionally an optimistic estimate of simple undo cost. Arrays span 128 synthetic states, not a million-node search arena.

Python small-operation rows use seven batches after warmup; full games use five batches of eight games. `generate_warm_*` has warm caches and reflects one phase-specific position. `copy_execute_midgame_first_action` disables outer validation but retains whatever validation the production helpers perform. `GameEncoder` timing is separate from the PR snapshot format. The profile is gathered separately from throughput timing. Wire round trips use a prebuilt payload and warmed child, excluding message construction and process startup.
