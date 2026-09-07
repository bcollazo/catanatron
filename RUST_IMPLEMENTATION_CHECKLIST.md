# Rust engine implementation progress

Execution instructions: [RUST_EXECUTION_GUIDE.md](RUST_EXECUTION_GUIDE.md). Architecture/evidence: [RUST_IMPLEMENTATION_PLAN.md](RUST_IMPLEMENTATION_PLAN.md).

Status at handoff: **planning complete; production implementation not started**. Existing executable code is limited to `experiments/rust-rollout/` design probes. The user will choose the implementation model. No remote branch or PR is requested.

## Ordered tasks

- [x] E00 — Checkout/toolchain/baseline provenance verified for the implementation session.
- [x] E01 — Production Cargo workspace builds; dependency direction enforced.
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

* Current task: E02 (freeze rules, topology, and fixtures; in progress).
* Last implementation check: topology and fixture export both pass their `--check` mode on 2026-09-07; targeted Python rule tests pass (24 tests), and `cargo check` still passes.
* Next action: add crafted city and domestic-trade transition fixtures (offer/accept/reject/confirm/cancel), then add forced chance fixtures and remaining phase/divergence coverage before marking E02 complete.
* Blocking condition: none for core work. PR #386 head `5149b1869ba6318a2f2e3ef3925915576a433286` is locally available but not merged into local `main`; real stdio certification remains pending that source or a recorded successor.
* Changed implementation files: `rust/` workspace files, generated topology fixtures/tables, `rust/docs/provenance.md`, `rust/docs/rules-profile.md`, this checklist.
* Known failing fixture/test IDs: no failing tests. E02's manifest deliberately reports incomplete coverage: `BUILD_CITY`, `OFFER_TRADE`, `ACCEPT_TRADE`, `REJECT_TRADE`, `CONFIRM_TRADE`, and `CANCEL_TRADE` are not yet represented by sampled traces.
* Decisions since the plan: use the isolated ignored `.venv/rust-engine` (Python 3.12.14, NetworkX 3.5) because the system pyenv shim is unconfigured. Topology exports only immutable geometry: resource/port assignments are deliberately excluded because Python map initialization randomizes them. Trace action selection is canonicalized because Python set-backed move menus otherwise vary by hash seed.

### 2026-09-07 — E02 checkpoint

* Added the `rust-v1` profile and v1 fixture-format specification.
* Added deterministic topology and fixture exporters. `export_topology.py --check` and `export_fixtures.py --check` pass. The fixture manifest records hashes and actual sampled coverage instead of pretending the sample corpus is complete.
* Sample traces cover BASE for 2/3/4 players and TOURNAMENT for 4 players, but not yet all actions/phases; E02 remains open.

### 2026-09-07 — E00 complete

* Commit at start: `5cc406a4032b3a7f6971b4fc0735663d2e0ca7e5` on `plan/rust-rollout-engine`.
* Recorded checkout, pinned rules/protocol revisions, available tool versions, Python environment, and commands in `rust/docs/provenance.md`.
* The offline Rust design probe and the Python baseline both passed. CPU WMI inspection was denied and is recorded as unavailable rather than assumed.

### 2026-09-07 — E01 complete

* Created the dependency-free `catanatron-core` workspace member with safe-Rust policy, release settings, committed lockfile, and a runnable workspace README.
* `cargo check`, formatting, and unit tests passed. No production crate depends on Python, JSON, or search code; the planning probe remains outside the workspace.

Append a dated entry after each completed task/session with commit, commands actually run, outcomes, benchmark report paths and precise next step. Keep incomplete/blocked/missed gates unchecked.
