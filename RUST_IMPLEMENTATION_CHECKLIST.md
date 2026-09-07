# Rust engine implementation progress

Execution instructions: [RUST_EXECUTION_GUIDE.md](RUST_EXECUTION_GUIDE.md). Architecture/evidence: [RUST_IMPLEMENTATION_PLAN.md](RUST_IMPLEMENTATION_PLAN.md).

Status at handoff: **planning complete; production implementation not started**. Existing executable code is limited to `experiments/rust-rollout/` design probes. The user will choose the implementation model. No remote branch or PR is requested.

## Ordered tasks

- [ ] E00 — Checkout/toolchain/baseline provenance verified for the implementation session.
- [ ] E01 — Production Cargo workspace builds; dependency direction enforced.
- [ ] E02 — Rules profile, full fixtures, topology export and named divergences frozen.
- [ ] E03 — Typed state/action/phase/outcomes and atomic checked boundaries implemented.
- [ ] E04 — Setup, construction, ordinary turns and independent move generation pass.
- [ ] E05 — Dice, production, discards, robber and chance outcomes pass.
- [ ] E06 — All development cards, awards and domestic trades pass.
- [ ] E07 — Complete games, RNG streams, random/weighted policies and scalar rollouts pass.
- [ ] E08 — Differential harness, benchmarks and allocation report implemented.
- [ ] E09 — Stdio adapter/importer and real host certification pass.
- [ ] E10 — Bounded flat Monte Carlo search bot works and is evaluated.
- [ ] E11 — Parallel results match scalar execution; scaling measured.
- [ ] E12 — MINI and batched Python extension pass parity/lifetime/error tests.
- [ ] E13 — Final checks, docs, examples and handoff complete.

## Independent release gates

- [ ] Functional: complete supported games/actions/configurations, no unfinished stubs.
- [ ] Correctness: zero unexplained differences; all named divergences independently tested.
- [ ] Protocol: 100 real mixed-process games with zero unexpected fallback/timeout/illegal action.
- [ ] RL interface: scalar/batch parity, documented action/feature versions, safe array lifetimes.
- [ ] Allocations: zero warmed baseline rollout allocations across supported phases.
- [ ] Speed: >=10x comparable same-machine Python games and fixed-state rollout throughput.
- [ ] Speed objective: >=1M player intents/second on one core, with workload specified.
- [ ] Competitive claim: comparable external-engine results support any “fastest” wording (follow-up; not required to ship functional v1).

## Current execution checkpoint

* Current task: E00 (not started).
* Last implementation check: none; only the committed planning experiments were run.
* Next action: inspect checkout/tools, record provenance, run the existing offline Rust probe.
* Blocking condition: none known for beginning core implementation; real stdio certification needs the pinned/merged PR host.
* Changed implementation files: none.
* Known failing fixture/test IDs: none generated yet.
* Decisions since the plan: none.

Append a dated entry after each completed task/session with commit, commands actually run, outcomes, benchmark report paths and precise next step. Keep incomplete/blocked/missed gates unchecked.
