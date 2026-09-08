# Catanatron Rust rollout engine

This directory is the production Rust workspace for the `rust-v1` Catanatron
rules profile. It is being implemented in stages described by
[`../RUST_EXECUTION_GUIDE.md`](../RUST_EXECUTION_GUIDE.md). The initial
workspace contains only `catanatron-core`, a safe, standard-library-only rules
kernel; search, stdio, benchmark, and Python-extension packages will be added
when their implementation stages begin.

## Local checks

From the repository root:

```powershell
cargo check --manifest-path rust/Cargo.toml
cargo fmt --check --manifest-path rust/Cargo.toml
cargo test --manifest-path rust/Cargo.toml
```

The workspace intentionally has no default dependency on Python, JSON,
networking, or search code. That keeps rule transitions independently usable
and testable.

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
