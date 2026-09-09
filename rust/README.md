# Catanatron Rust rollout engine

This directory is the production Rust workspace for the `rust-v1` Catanatron
rules profile. It was implemented in stages described by
[`../RUST_EXECUTION_GUIDE.md`](../RUST_EXECUTION_GUIDE.md). The initial
release contains a safe, dependency-free `catanatron-core` rules kernel,
scalar and parallel rollout search, conformance/benchmark tooling, a v1 JSONL
bot, and an optional batched Python extension.

## Supported scope

The engine supports 2–4 players, BASE and MINI maps, and the fixed TOURNAMENT
assignment. It implements the complete `rust-v1` action/rule profile and uses
perfect-information state. Hidden-state belief sampling, fair-information
observations, RL training, neural inference, and competitive “fastest engine”
claims are outside this release.

## Local checks

From the repository root:

```powershell
cargo fmt --all --manifest-path rust/Cargo.toml -- --check
cargo clippy --workspace --all-targets --manifest-path rust/Cargo.toml -- -D warnings
cargo test --workspace --manifest-path rust/Cargo.toml
```

The core package has no dependency on Python, JSON, networking, or search
code. Protocol concerns remain in the bot package.

## In-process Rust

Add `catanatron-core` and `catanatron-search` as path dependencies, then:

```rust
use catanatron_search::{initialize_base, rollout, NumberPlacement, Policy, RolloutLimits, RolloutScratch};

let (context, root) = initialize_base(4, NumberPlacement::OfficialSpiral, 7, 0)?;
let result = rollout(&context, &root, Policy::Weighted, 99, RolloutLimits::default(), &mut RolloutScratch::default());
println!("winner={:?}, turns={}", result.winner, result.turns);
# Ok::<(), catanatron_core::IllegalAction>(())
```

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
--seed 7 --threads 1`. The stdio search adapter remains single-threaded;
parallel fixed-root rollouts are available through the Rust/Python batch APIs. To repeat its host
gate:

```powershell
.\.venv\rust-engine\Scripts\python.exe rust\tools\verify_stdio.py --bot rust\target\release\catanatron-bot.exe --games 100 --host-worktree C:\dev\catanatron-pr386
```

## Optional Python batch package

Build this separately so the repository's existing Python package metadata is
unchanged:

```powershell
python -m pip install "maturin>=1,<2" numpy
python -m maturin build --release --manifest-path rust/crates/python/Cargo.toml --out rust/target/wheels
python -m pip install --force-reinstall rust/target/wheels/catanatron_rust-*.whl
```

```python
from catanatron_rust import Batch

batch = Batch(16, players=4, map="BASE", seed=7)
view = batch.observe_many(list(range(16)))
first = int(view["action_ids"][view["menu_offsets"][0]])
step = batch.step_many([0], [first])
batch.reset_many([0], [99])  # reset is always explicit
```

See [`docs/features-v1.md`](docs/features-v1.md) for exact shapes and ownership,
and [`crates/python/README.md`](crates/python/README.md) for the Gym catalogue
adapter and reward contract.

## Conformance and benchmarks

```powershell
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench --bin catanatron-conformance -- rust/tests/fixtures/transitions/sample-base-2p.jsonl
.\.venv\rust-engine\Scripts\python.exe rust\tools\differential.py --fixtures rust\tests\fixtures --games-per-config 100
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- games --games 100 --players 4 --map BASE --policy random
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- rollouts --fixtures rust/tests/fixtures --rollouts 1000 --players 4 --map BASE --policy weighted
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- allocations --fixtures rust/tests/fixtures --rollouts 100 --players 4 --map BASE --policy weighted
```

See [`docs/performance.md`](docs/performance.md) for measured results and
comparison constraints, [`docs/rules-profile.md`](docs/rules-profile.md) for
the exact rules, and [`docs/python-divergences.md`](docs/python-divergences.md)
for the five independently tested Python inconsistencies intentionally corrected
by `rust-v1`.

## Design probes are separate

`experiments/rust-rollout/` is a committed, standalone planning probe. It is
not a member of this workspace and is not the production engine. Run it
separately after its dependencies have been cached:

```powershell
cargo run --release --offline --manifest-path experiments/rust-rollout/Cargo.toml
```

See [`docs/provenance.md`](docs/provenance.md) for the implementation-session
baseline and commands that have actually run.
