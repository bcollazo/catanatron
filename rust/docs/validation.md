# Rust v1 validation record

Validation date: 2026-09-08. Branch: `plan/rust-rollout-engine`.

## Independent statuses

| Area | Status | Evidence |
|---|---|---|
| Functional readiness | Ready for the documented v1 scope | Workspace release tests; BASE/MINI/TOURNAMENT initialization and complete-game matrices |
| Correctness | Passed with five intentional, documented Python divergences | 275,514 live transitions in 300 games; 273,438 exact matches, 2,076 recognized divergence occurrences, zero unexplained failures |
| Protocol | Certified | 100 games against protocol/schema v1 at Python host revision `5149b1869ba6318a2f2e3ef3925915576a433286`; zero timeout, illegal action, fallback, or mismatch |
| Performance targets | Met | Final BASE games: 1.070M player intents/s; fixed-root rollouts: 0.977M intents/s; warmed rollouts: zero allocations |
| Competitive “fastest” claim | Not established | No comparable current external-engine tournament or benchmark was run |

The live differential count treats occurrences of D001–D005 as recognized differences, not exact
matches. Definitions and minimal tests are in [`python-divergences.md`](python-divergences.md).

## Final artifacts

* `rust/bench-results/final-differential.json`: 300 live Python games across 2/3/4 players.
* `rust/bench-results/final-games.json`: five batches of 1,000 newly initialized games.
* `rust/bench-results/final-rollouts.json`: five batches of 10,000 fixed-root rollouts.
* `rust/bench-results/final-allocations.json`: 1,000 warmed fixed-root rollouts.
* `rust/bench-results/e09-stdio.json`: original 100-game protocol certification; E13 repeated it with the same all-green result.
* [`features-v1.md`](features-v1.md): exact Python observation/action schema.
* [`performance.md`](performance.md): workload definitions, comparisons, and limitations.

## Reproduction

The command matrix is maintained in E13 of
[`../../RUST_EXECUTION_GUIDE.md`](../../RUST_EXECUTION_GUIDE.md). Generated topology, transition,
and Gym catalogue checks are deterministic and non-mutating under `--check`.
