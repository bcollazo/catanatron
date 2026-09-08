# E08 performance scoreboard

Measured 2026-09-08 on the implementation host described in
`provenance.md`. All Rust measurements use the release workspace profile,
one thread, portable compiler flags, five raw timing batches, BASE, four
players, and seed 8600.

| Workload | Python intents/s | Rust intents/s | Ratio |
|---|---:|---:|---:|
| 40 random complete games (5 × 8) | 68,732 | 1,042,729 | 15.17× |
| 40 random fixed-root rollouts (5 × 8) | 71,264 | 983,890 | 13.81× |
| 5,000 weighted fixed-root rollouts | — | 1,074,932 | — |
| 500 weighted complete games | — | 1,142,688 | — |

The matched comparisons use the same player count, BASE map, random policy,
seed, number of games and batch structure. Chance generators are native to
each engine; correctness is established separately by forced-outcome
differential testing. The fixed-root workload excludes root initialization.
All games completed without truncation.

The first Rust measurement was 74,078 intents/s. Inspection showed that exact
longest-road DFS was refreshed after every intent and then repeated by the
context-aware wrapper. One controlled change limited award refresh to roads,
settlements and Knight plays and removed the duplicate wrapper refresh. The
identical 500-game weighted workload then reached 1,142,688 intents/s (15.43×
the original Rust result). Release tests and differential goldens pass after
the change.

The warmed allocation region covers rollout execution only, after scratch
construction and one warmup rollout. Across 100 weighted fixed-root rollouts
(65,555 player intents), it recorded zero allocations, zero deallocations and
zero allocated bytes. `Position` is 232 bytes.

Selected raw JSON reports are under `rust/bench-results/2026-09-08/`.
These measurements establish the project targets, not a competitive
world-record claim.

## E11 parallel fixed-root rollouts

The host reports 32 logical processors. Physical-core count and SMT topology
could not be read, so the table reports requested software worker counts only.
Each cell is aggregate player intents/second across five deterministic batches;
all thread counts produced exactly the scalar per-rollout results.

| Batch size | 1 worker | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.95M | 1.86M | 3.00M | 4.73M | 5.95M | 7.05M |
| 2,048 | 0.98M | 1.94M | 3.80M | 7.08M | 10.08M | 14.92M |

The small batch shows thread-start and aggregation overhead more strongly. The
large batch reaches 15.17x its measured one-worker throughput at 32 workers;
this is observed scaling, not a claim of linear scaling or physical-core
efficiency. The estimated caller-owned root/result payload is 7,808 bytes for
32 roots and 499,712 bytes for 2,048 roots. Process peak RSS and per-thread
stack reservation were not measured by the portable harness and are therefore
not represented as memory measurements. Raw reports are the
`parallel-<batch>-t<workers>.json` files in the dated results directory.

## E12 Python batch crossing

The optional CPython 3.12 abi3 wheel was built in release mode and timed with 256 BASE
environments on the same 32-logical-processor Windows host. The committed timing script includes
NumPy result creation and the Python/Rust boundary:

| Operation | Throughput |
|---|---:|
| Eight-worker complete rollouts | 10,504 rollouts/s |
| `observe_many` plus menus | 90,877 environments/s |
| `reset_many` plus observations/menus | 117,113 environments/s |

These are single-run engineering measurements, not cross-machine claims. Observation and reset
figures amortize one Python call over 256 environments.
