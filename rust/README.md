# Catanatron Rust rollout engine

This directory is the production Rust workspace for the `rust-v1` Catanatron
rules profile. It is being implemented in stages described by
[`../RUST_EXECUTION_GUIDE.md`](../RUST_EXECUTION_GUIDE.md). The initial
workspace contains a safe, dependency-free `catanatron-core` rules kernel,
scalar rollout search, conformance/benchmark tooling, and the v1 JSONL bot.

## Local checks

From the repository root:

```powershell
cargo check --manifest-path rust/Cargo.toml
cargo fmt --check --manifest-path rust/Cargo.toml
cargo test --manifest-path rust/Cargo.toml
```

The core package has no dependency on Python, JSON, networking, or search
code. Protocol concerns remain in the bot package.

## Stdio bot

Build the bot and register the resulting executable with the Python host:

```powershell
cargo build --release --manifest-path rust/Cargo.toml -p catanatron-bot
catanatron-play --bot "RUST=exec:C:/dev/catanatron/rust/target/release/catanatron-bot.exe" --players RUST,R,R,R --num 10
```

The bot imports each full dynamic snapshot and verifies exact semantic parity
with the host's offered action menu. It implements protocol/schema v1 from
pinned PR #386. Random is the default policy; bounded rollout search is enabled
with, for example, `--policy rollout --simulations 10000 --budget-ms 100
--seed 7 --threads 1`. E11 will add parallel thread counts. To repeat its host
gate:

```powershell
.\.venv\rust-engine\Scripts\python.exe rust\tools\verify_stdio.py --bot rust\target\release\catanatron-bot.exe --games 100 --host-worktree C:\dev\catanatron-pr386
```

## Conformance and benchmarks

```powershell
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench --bin catanatron-conformance -- rust/tests/fixtures/transitions/sample-base-2p.jsonl
.\.venv\rust-engine\Scripts\python.exe rust\tools\differential.py --games 100 --players 2 3 4
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- games --games 100 --players 4 --policy random
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- rollouts --fixtures 1 --rollouts 1000 --players 4 --policy weighted
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- allocations --fixtures 1 --rollouts 100 --players 4 --policy weighted
```

See [`docs/performance.md`](docs/performance.md) for measured results and
comparison constraints.

## Design probes are separate

`experiments/rust-rollout/` is a committed, standalone planning probe. It is
not a member of this workspace and is not the production engine. Run it
separately after its dependencies have been cached:

```powershell
cargo run --release --offline --manifest-path experiments/rust-rollout/Cargo.toml
```

See [`docs/provenance.md`](docs/provenance.md) for the implementation-session
baseline and commands that have actually run.
