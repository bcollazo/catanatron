# Rust engine implementation progress

Execution instructions: [RUST_EXECUTION_GUIDE.md](RUST_EXECUTION_GUIDE.md). Architecture/evidence: [RUST_IMPLEMENTATION_PLAN.md](RUST_IMPLEMENTATION_PLAN.md).

Status at handoff: **planning complete; production implementation not started**. Existing executable code is limited to `experiments/rust-rollout/` design probes. The user will choose the implementation model. No remote branch or PR is requested.

## Ordered tasks

- [x] E00 — Checkout/toolchain/baseline provenance verified for the implementation session.
- [x] E01 — Production Cargo workspace builds; dependency direction enforced.
- [x] E02 — Rules profile, full fixtures, topology export and named divergences frozen.
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

* Current task: E03 (typed state, actions, phases, and checked boundaries).
* Last implementation check: typed-model invariants pass: `cargo test --manifest-path rust/Cargo.toml` (5 tests) and formatting checks pass. Recorded layout is `Position=210B`, `Action=11B`.
* Next action: extend checked validation/application with action-specific resource, inventory, geometry, and outcome errors while retaining failure atomicity; then decide E03 completion.
* Blocking condition: none for core work. PR #386 head `5149b1869ba6318a2f2e3ef3925915576a433286` is locally available but not merged into local `main`; real stdio certification remains pending that source or a recorded successor.
* Changed implementation files: `rust/` workspace files, generated topology/transition fixtures/tables, `rust/docs/provenance.md`, `rust/docs/rules-profile.md`, this checklist.
* Known failing fixture/test IDs: none. E02 complete; later Rust conformance tests must consume the committed corpus.
* Decisions since the plan: use the isolated ignored `.venv/rust-engine` (Python 3.12.14, NetworkX 3.5) because the system pyenv shim is unconfigured. Topology exports only immutable geometry: resource/port assignments are deliberately excluded because Python map initialization randomizes them. Trace action selection is canonicalized because Python set-backed move menus otherwise vary by hash seed.

### 2026-09-07 — E02 checkpoint

* Added the `rust-v1` profile and v1 fixture-format specification.
* Added deterministic topology and fixture exporters. `export_topology.py --check` and `export_fixtures.py --check` pass. The fixture manifest records hashes and actual sampled coverage instead of pretending the sample corpus is complete.
* Sample traces cover BASE for 2/3/4 players and TOURNAMENT for 4 players, but not yet all actions/phases; E02 remains open.

### 2026-09-07 — E02 action-coverage checkpoint

* Added crafted city and domestic-trade transitions. The deterministic manifest now covers every `ActionType`, including offer/accept/reject/confirm/cancel trade paths.
* `export_fixtures.py --check` and the focused Python game/action/yield suite (45 tests) pass. Do not treat action-type coverage as full E02 completion: divergence and explicit chance/phase coverage remain outstanding.

### 2026-09-07 — E02 complete

* Added exact chance weight fixtures and the named, reproduced `D001-domestic-trade-proposer-revisited` compatibility correction.
* Audited the generated records: all 18 actions and each of the seven reachable Python decision prompts occur before and after transitions; BASE/TOURNAMENT and 2/3/4-seat coverage are present. Exporters are deterministic under `--check`.

### 2026-09-07 — E03 typed-model checkpoint

* Added private checked dense IDs, typed action/resource/development/phase/outcome/status definitions, fixed-array state, and non-mutating checked boundary validation.
* Invariant tests cover invalid IDs, wrong actor/phase, pending-chance and terminal rejection, copy independence, and the measured size budget. E03 remains open until the checked mutation API supplies all required action-specific atomic errors.

### 2026-09-07 — E00 complete

* Commit at start: `5cc406a4032b3a7f6971b4fc0735663d2e0ca7e5` on `plan/rust-rollout-engine`.
* Recorded checkout, pinned rules/protocol revisions, available tool versions, Python environment, and commands in `rust/docs/provenance.md`.
* The offline Rust design probe and the Python baseline both passed. CPU WMI inspection was denied and is recorded as unavailable rather than assumed.

### 2026-09-07 — E01 complete

* Created the dependency-free `catanatron-core` workspace member with safe-Rust policy, release settings, committed lockfile, and a runnable workspace README.
* `cargo check`, formatting, and unit tests passed. No production crate depends on Python, JSON, or search code; the planning probe remains outside the workspace.

### 2026-09-07 — E04 post-roll road checkpoint

* Added post-roll paid-road generation, checked geometry/resource/inventory validation, and atomic application. A generated road now pays wood and brick to the bank, consumes a road piece, and leaves the actor in post-roll.
* `cargo test --manifest-path rust/Cargo.toml` (12 tests) and `cargo fmt --check --manifest-path rust/Cargo.toml` pass.
* Next action: implement checked settlement construction and its post-roll legal-menu predicate; then expand E04 construction coverage.

### 2026-09-07 — E04 post-roll settlement checkpoint

* Added post-roll paid-settlement generation and atomic application. The checked path enforces empty/distance-legal vertices, an incident owned road, inventory, and wood/brick/sheep/wheat payment.
* `cargo test --manifest-path rust/Cargo.toml` (13 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: represent city building kind in canonical state, then implement checked paid city construction and generation.

### 2026-09-07 — E04 post-roll city checkpoint

* Added a compact city ownership encoding and post-roll city generation/application. A city requires an owned settlement, an available city piece, two wheat and three ore; it returns the upgraded settlement piece and pays resources to the bank.
* `cargo test --manifest-path rust/Cargo.toml` (14 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: implement paid development-card purchase and chance resolution; E04 construction actions will then have road, settlement, city, and development coverage.

### 2026-09-07 — E04 development-purchase checkpoint

* Added legal generation and atomic checked application for development-card purchases, including sheep/wheat/ore payment and an explicit pending draw. Added checked chance resolution for development draws, which updates the deck and player's typed card count before returning to post-roll.
* `cargo test --manifest-path rust/Cargo.toml` (15 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: implement setup's second-settlement resource grants and turn-counter behavior, then audit E04's ordinary-turn exit cases before moving to dice/robber work.

### 2026-09-07 — E04 setup-turn checkpoint

* Matched the pinned setup `advance_turn` counter semantics: forward setup-road advances and intermediate reverse advances increment the counter, while the turnaround and final setup road do not. The two-player snake regression asserts the resulting count.
* `cargo test --manifest-path rust/Cargo.toml` (15 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: add an explicit immutable layout/context layer before second-settlement payouts or dice production; the current generated topology deliberately contains geometry only, not the randomized resource/number assignment required by those rules.

### 2026-09-07 — topology incidence checkpoint

* Extended deterministic topology export with dense BASE/MINI land-tile-to-node tables and exposed typed Rust accessors. The exporter now formats its generated Rust with `rustfmt`, so `export_topology.py --check` and workspace formatting agree.
* `python rust/tools/export_topology.py --check`, `cargo test --manifest-path rust/Cargo.toml` (15 tests), and `cargo fmt --check --manifest-path rust/Cargo.toml` pass.
* Next action: add the immutable runtime layout/context carrying each land tile's resource and number token, then route setup payouts and dice production through that context.

### 2026-09-07 — layout and setup-payout checkpoint

* Added immutable validated land-tile layout data and `GameContext`, keeping randomized assignments outside copied `Position` values. Added a context-aware checked transition that grants each adjacent non-desert resource only to a player's second setup settlement.
* `cargo test --manifest-path rust/Cargo.toml` (17 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: route dice outcome resolution through this context, beginning with non-seven aggregate production and the specified per-resource bank-shortage rule.

### 2026-09-07 — E05 non-seven production checkpoint

* Added context-aware non-seven dice resolution. It aggregates settlement/city demand tile-locally, excludes the robber tile, and pays each resource only when the bank can satisfy aggregate demand; insufficient resources pay nobody.
* `cargo test --manifest-path rust/Cargo.toml` (18 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: implement seven handling: ordered discards, then robber menus and theft chance outcomes.

### 2026-09-07 — E05 seven-discard checkpoint

* Seven now enters the lowest qualifying seat's discard phase, requires exactly half of each hand above seven one card at a time, advances through higher qualifying seats, and restores the turn owner for robber selection. Discards return cards to the bank atomically.
* `cargo test --manifest-path rust/Cargo.toml` (19 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: generate and apply robber destinations/victim choices, then resolve theft through its pending chance outcome.

### 2026-09-07 — E05 robber/theft checkpoint

* Added deterministic robber menus for every non-current tile, one action per distinct eligible victim, victimless moves only when no eligible victim exists, and explicit theft chance resolution. Checked failures remain atomic.
* `cargo test --manifest-path rust/Cargo.toml` (20 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: add friendly-robber configuration/fallback coverage and audit E05 resume semantics before marking E05 complete.

### 2026-09-07 — E06 Knight/resume checkpoint

* Theft chance now preserves whether robber resolution must resume pre-roll or post-roll. Added end-turn development eligibility refresh plus checked Knight generation/application, card consumption, played-card limit, and played-knight count.
* `cargo test --manifest-path rust/Cargo.toml` (21 tests), `cargo fmt --check --manifest-path rust/Cargo.toml`, and `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` pass.
* Next action: implement Year of Plenty, Monopoly, and Road Building before awards/trading.

### 2026-09-07 — E06 Year-of-Plenty/Monopoly checkpoint

* Added pinned Year of Plenty pair/single menu generation and resource transfer, plus Monopoly menus and transfers from every opponent. Both use the shared eligibility/card-consumption/one-card-per-turn rules.
* `cargo test --manifest-path rust/Cargo.toml` (22 tests), formatting, and Clippy with warnings denied pass.
* Next action: implement Road Building's free-road phase, including early exhaustion and pre/post-roll resume.

### 2026-09-07 — E06 Road Building checkpoint

* Added Road Building eligibility/menu generation and a typed free-road phase. It places up to two geometry-legal roads without payment, stops if pieces or placements are exhausted, and resumes pre-roll/post-roll correctly.
* `cargo test --manifest-path rust/Cargo.toml` (23 tests), formatting, and Clippy with warnings denied pass.
* Next action: add maritime trading and then domestic trade state/response flows.

### 2026-09-07 — E06 domestic-trade checkpoint

* Added canonical pending-trade state and checked offer/accept/reject/confirm/cancel transitions. Each other seat is asked once in ascending order; confirmation revalidates both balances and transfers atomically.
* `cargo test --manifest-path rust/Cargo.toml` (24 tests), formatting, and Clippy with warnings denied pass.
* Next action: add port-aware maritime trade generation/application, then awards and victory maintenance.

### 2026-09-07 — E06 maritime-trade checkpoint

* Added immutable runtime port assignments, ownership-derived best rates (4:1/3:1/resource-specific 2:1), context-aware legal generation, and atomic checked exchange with bank availability validation.
* `cargo test --manifest-path rust/Cargo.toml` (25 tests), formatting, and Clippy with warnings denied pass.
* Next action: implement exact longest-road/largest-army maintenance and victory status.

Append a dated entry after each completed task/session with commit, commands actually run, outcomes, benchmark report paths and precise next step. Keep incomplete/blocked/missed gates unchecked.
