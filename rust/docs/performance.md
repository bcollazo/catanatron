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
