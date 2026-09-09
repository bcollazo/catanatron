# Rust engine implementation progress

Execution instructions: [RUST_EXECUTION_GUIDE.md](RUST_EXECUTION_GUIDE.md). Architecture/evidence: [RUST_IMPLEMENTATION_PLAN.md](RUST_IMPLEMENTATION_PLAN.md).

Status at handoff: **planning complete; production implementation not started**. Existing executable code is limited to `experiments/rust-rollout/` design probes. The user will choose the implementation model. No remote branch or PR is requested.

## Ordered tasks

- [x] E00 — Checkout/toolchain/baseline provenance verified for the implementation session.
- [x] E01 — Production Cargo workspace builds; dependency direction enforced.
- [x] E02 — Rules profile, full fixtures, topology export and named divergences frozen.
- [x] E03 — Typed state/action/phase/outcomes and atomic checked boundaries implemented.
- [x] E04 — Setup, construction, ordinary turns and independent move generation pass.
- [x] E05 — Dice, production, discards, robber and chance outcomes pass.
- [x] E06 — All development cards, awards and domestic trades pass.
- [x] E07 — Complete games, RNG streams, random/weighted policies and scalar rollouts pass.
- [x] E08 — Differential harness, benchmarks and allocation report implemented.
- [x] E09 — Stdio adapter/importer and real host certification pass.
- [x] E10 — Bounded flat Monte Carlo search bot works and is evaluated.
- [x] E11 — Parallel results match scalar execution; scaling measured.
- [x] E12 — MINI and batched Python extension pass parity/lifetime/error tests.
- [x] E13 — Final checks, docs, examples and handoff complete.

## Independent release gates

- [x] Functional: complete supported games/actions/configurations, no unfinished stubs.
- [x] Correctness: zero unexplained differences; all named divergences independently tested.
- [x] Protocol: 100 real mixed-process games with zero unexpected fallback/timeout/illegal action.
- [x] RL interface: scalar/batch parity, documented action/feature versions, safe array lifetimes.
- [x] Allocations: zero warmed baseline rollout allocations across supported phases.
- [x] Speed: >=10x comparable same-machine Python games and fixed-state rollout throughput.
- [x] Speed objective: >=1M player intents/second on one core, with workload specified.
- [ ] Competitive claim: comparable external-engine results support any “fastest” wording (follow-up; not required to ship functional v1).

## Current execution checkpoint

* Current task: complete through E13.
* Last implementation check: the E13 release matrix passed, including 275,514 live differential transitions, 100 stdio host games, all workspace release tests and six wheel-level Python tests. Final benchmark and allocation reports are committed.
* Next action: optional follow-up optimization or external competitive evaluation; neither blocks functional v1.
* Blocking condition: none. Protocol v1/schema v1 were certified against PR #386 head `5149b1869ba6318a2f2e3ef3925915576a433286` in the isolated `C:\dev\catanatron-pr386` worktree.
* Changed implementation files: `rust/` workspace files, generated topology/transition fixtures/tables, `rust/docs/provenance.md`, `rust/docs/rules-profile.md`, this checklist.
* Known failing fixture/test IDs: none. E02 complete; later Rust conformance tests must consume the committed corpus.
* Decisions since the plan: use the isolated ignored `.venv/rust-engine` (Python 3.12.14, NetworkX 3.5) because the system pyenv shim is unconfigured. Topology exports only immutable geometry: resource/port assignments are deliberately excluded because Python map initialization randomizes them. Trace action selection is canonicalized because Python set-backed move menus otherwise vary by hash seed.

### 2026-09-08 — E09 complete

* Added the `catanatron-bot` JSONL process boundary. It defaults to `observe=false`, replies only to `hello`/`decide`, flushes each reply, exits cleanly on EOF, ignores unknown notifications, and rejects protocol, schema, game, color, numeric, duplicate-road, and malformed-action errors clearly on stderr.
* The importer caches only the static map from `before`, refreshes all dynamic fields on every `decide`, preserves wire seat order, eligibility and award incumbents, derives setup's latest settlement from ordered `buildings_by_color`, treats `current_trade[10]` as a seat index, and labels no hidden-state inference beyond the perfect-information v1 snapshot.
* Rust generation is compared semantically with the entire offered host menu before selection; the random policy returns the original offered wire triple. Unit coverage includes all 18 action payload types and process lifecycle/error cases.
* `rust/tools/verify_stdio.py` ran 100 games against pinned PR #386 across 2/3/4-player and multiple Rust seat schedules: 100 completed, zero unexpected fallbacks, zero timeouts, zero illegal actions, and zero root-menu mismatches. Report: `rust/bench-results/e09-stdio.json`.

### 2026-09-08 — E10 complete

* Added round-robin flat Monte Carlo over every offered root action. Each simulation copies the root, applies one intent, samples immediate chance, and uses the weighted policy to terminal/cutoff. Root-player rewards are 1 win, 0 loss and 0.5 cutoff; deterministic action-key ties and fixed-seed tests are included.
* Added `--policy random|rollout`, `--simulations`, `--budget-ms`, `--seed`, and `--threads`. Default deadline is 100 ms with 5 ms reserved for response serialization; deadline checks occur between simulations and inside rollouts at every action/chance boundary. E10 rejects thread counts above one until E11.
* Forced-win, cutoff-over-loss reward, legal-selection, deterministic fixed-seed, deadline interruption, and parent-root immutability regressions pass.
* The fixed evaluation at 20 ms covered 20 seat-rotated games and 5,569 decisions. Of 2,542 searched decisions, rollouts averaged 204.96 (median 129.5, range 18–10,000). Latency was p50 0.0014 ms, p95 16.75 ms, p99 17.33 ms, max 24.66 ms. Observed records were 9/10 versus Random and 10/10 versus WeightedRandom; the report explicitly does not claim superiority from this small sample. Raw report: `rust/bench-results/e10-search.json`.

### 2026-09-08 — E11 complete

* Added `Batch` and `rollout_many` with fixed worker chunks, worker-local position/scratch/RNG state, deterministic input-index aggregation, and whole-batch validation before work starts.
* Scalar and 1/2/4-thread results match exactly for fixed seeds; mixed/disjoint roots remain unchanged and cannot corrupt each other.
* On the 32-logical-processor host, the 2,048-root batch measured 0.98M, 1.94M, 3.80M, 7.08M, 10.08M and 14.92M intents/s at 1/2/4/8/16/32 requested workers. The 32-root batch measured 0.95M through 7.05M over the same range. Physical-core and SMT topology are unavailable, so these are not labeled physical-core measurements.
* Estimated root/result payload memory was 7,808 bytes for 32 roots and 499,712 bytes for 2,048. Peak RSS and stack reservations were not measured and remain explicitly unavailable. Raw reports are committed under `rust/bench-results/2026-09-08/parallel-*.json`.

### 2026-09-08 — E12 complete

* Added map-aware BASE/MINI geometry with generated active node/edge/tile masks, MINI initialization and 2/3/4-player scalar games. Three new Python trace corpora pass exact state, action-menu and transition differential checks.
* Added the fixed Python TOURNAMENT tile, number, desert and port assignment on BASE topology.
* Added the optional `catanatron-rust` abi3 wheel with owned NumPy observations, explicit batch reset/step/observe/rollout calls, interpreter detachment, atomic validation, independent chance streams, stable rewards/truncation, and immutable rollout roots.
* Added versioned perfect-information features and generation-stamped dynamic action IDs. Exported all nine Python Gym catalogues; setup legality and catalogue sizes agree exactly for BASE/MINI/TOURNAMENT with 2/3/4 seats.
* Six wheel-level tests cover stale views/IDs, atomic failures, reset semantics, Gym mapping, and fixed-seed equality for batches 1/16/256 across 1/2/4 workers. The 256-environment timing measured 10,504 complete rollouts/s, 90,877 observations/s, and 117,113 resets/s including Python/NumPy crossing costs.

### 2026-09-08 — E13 complete

* Added final benchmark/differential CLI contracts, BASE/MINI/TOURNAMENT benchmark initialization, a focused Rust/fixture/extension CI job, complete installation/examples, and a validation evidence index.
* The final BASE differential run covered 300 games and 275,514 transitions: 273,438 exact matches, 2,076 recognized D001–D005 occurrences, and zero unexplained failures.
* Repeated stdio certification against host revision `5149b1869ba6318a2f2e3ef3925915576a433286`: 100/100 games, zero timeout, illegal action, fallback, or protocol/schema mismatch.
* Final one-worker Random measurements: 1,070,247 intents/s across 5,000 newly initialized games; 977,378 intents/s across 50,000 fixed-root rollouts; zero allocations across 1,000 warmed rollouts.
* Functional readiness, correctness, protocol certification and measured speed are green. No external competitive “fastest” claim is made.

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

### 2026-09-07 — E03–E07 complete; E08 active

* Completed typed atomic boundaries; setup and ordinary construction; dice/discard/robber/chance rules; all development cards, maritime/domestic trades, awards and victory; reproducible initialization/RNG/policies; and root-preserving scalar rollout.
* The release game matrix completed 400/400 games with winners and zero turn-limit/action-limit truncations: 100 seeds for each 2/4-player × random/weighted combination.
* Passed: workspace formatting; Clippy with warnings denied; release core/search tests; deterministic topology export; deterministic fixture export covering all 18 action variants.
* Current task is E08. Protocol certification, differential equality, allocation targets, and speed targets remain open and are not implied by functional E03–E07 completion.

### 2026-09-07 — E08 canonical-import checkpoint

* Added the `catanatron-bench` workspace package and a serde-only fixture boundary; core remains independent of JSON and Python.
* Upgraded canonical fixtures to version 2 so snapshots explicitly carry typed phase/resume payloads, owned roads, ports, award state, and domestic-trade response progress instead of asking Rust to infer missing facts.
* Every before/after boundary in all five committed transition corpora deserializes and imports into typed Rust context/state. Deterministic fixture regeneration, focused tests, formatting, and Clippy with warnings denied pass.
* This checkpoint proves lossless import coverage only. Transition/menu parity is the next gate and is not yet claimed.

### 2026-09-07 — E08 golden-transition parity checkpoint

* Added the `catanatron-conformance` JSONL executable. It imports each before state, compares the pre-action legal menu, applies the typed intent and forced outcome, and compares after-state, status, and post-action legal menu with first-field mismatch reporting.
* All 394 committed Python transition records are equal under `rust-v1`; the release runner reports `{"cases":394,"status":"equal"}`.
* Differential work found and corrected two exporter ambiguities (stale Python longest-road and robber flags) and aligned Rust's eligibility mask with Python's turn-start snapshot semantics. The full workspace suite (38 tests), formatting, and Clippy with warnings denied pass.
* This is golden-corpus parity, not the E08 exit: live controlled full-game trajectories, divergence accounting, benchmark reports, and allocation measurement remain open.

### 2026-09-08 — E08 live differential checkpoint

* Added the live Python/Rust differential driver with canonical action selection, forced Python outcomes, terminal-menu normalization, first-unexpected capture, and exact known-divergence accounting.
* Fixed a Rust correctness bug found by the live run: development-card/theft chance completion now executes award/victory finalization; a victory-point draw that reaches 10 immediately returns `Won` and enters `Terminal`.
* The required 100 games for each 2/3/4-player BASE configuration completed 300/300 with 275,514 checked transitions, zero truncations, and zero unexplained failures. 176 games were fully equal; 124 encountered one of the narrowly registered Python longest-road corrections and were counted as divergent rather than equal.
* Registered D002–D005 for pinned Python longest-road undercounts, incumbent-tie transfer, and below-threshold award behavior. Each allowance requires every non-award field to match; terminal differences are accepted only when fully caused by that award delta.
* Report: `rust/bench-results/2026-09-08/differential-100x-2p-3p-4p.json`. At this checkpoint E08 still remained open for performance and allocation gates; those are closed below.

### 2026-09-08 — E08 complete

* Added `catanatron-bench` games, fixed-root rollouts, kernels, and warmed-allocation subcommands with strict CLI errors and JSON reports containing five raw timing samples.
* One controlled optimization removed redundant/unnecessary award recomputation. Weighted complete-game throughput improved from 74,078 to 1,142,688 intents/s; weighted fixed-root throughput is 1,074,932 intents/s.
* Matched random-policy comparisons measured 15.17× Python throughput for complete games and 13.81× for fixed-root rollouts. The warmed 100-rollout region recorded zero allocations/deallocations/bytes.
* Release tests, Clippy with warnings denied, formatting, fixture determinism, golden conformance, and the live 300-game differential corpus pass. E08 is complete; E09 stdio integration is next.

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

### 2026-09-07 — E06 awards/victory checkpoint

* Added exact edge-simple longest-road DFS with opponent-building stops, incumbent tie retention/removal, Largest Army selection, derived actual VP, last-qualifying-seat winner scans, and winner-before-turn-limit terminal precedence.
* `cargo test --manifest-path rust/Cargo.toml` (26 tests), formatting, and Clippy with warnings denied pass.
* Next action: broaden E05/E06 matrix coverage (chance weights, award ties/blocks, proposer seats, failed confirmation) and remove the remaining unsupported action paths before task completion.

### 2026-09-07 — E05 chance/friendly-robber checkpoint

* Added exact integer chance enumeration: all 36 concrete dice pairs, theft weights from victim holdings, and development draw weights from deck counts. Added context-configured friendly-robber filtering with unfiltered fallback behavior.
* `cargo test --manifest-path rust/Cargo.toml` (28 tests), formatting, and Clippy with warnings denied pass.
* Next action: consume the exact chance table from E07's deterministic sampler and complete the remaining E05/E06 matrix tests while exercising full games.

### 2026-09-07 — E07 RNG/initialization/policy checkpoint

* Added `catanatron-search` with pinned ChaCha8Rng, rejection-sampled bounded draws, stable separated child seeds with known-answer vectors, exact chance sampling, and random/weighted cumulative action policies.
* Added reproducible BASE initialization with shuffled standard resources and ports plus `official_spiral` and `random` number placement. The desert establishes the initial robber tile.
* Workspace tests (31 total), formatting, and Clippy with warnings denied pass.
* Next action: implement scalar rollout with separate chance/policy streams and explicit action/turn caps, then run deterministic 2/4-player game suites and conservation checks.

### 2026-09-07 — E07 scalar-rollout checkpoint

* Added root-copying scalar rollout with separate chance/policy streams, reusable action/outcome scratch, player-intent accounting, and distinct turn/action truncation. Debug trajectories assert resource and piece conservation after every transition.
* Release validation completed 100 seeds for every 2/4-player × random/weighted combination (400 games total) with no action-limit failures. Determinism and root independence tests pass.
* `cargo test --release --manifest-path rust/Cargo.toml -p catanatron-search --test full_games`, focused debug tests, formatting, and Clippy with warnings denied pass.
* Next action: close the remaining E05/E06 edge-case matrix and run the full workspace suite before marking E05–E07 complete and handing off at E08.

Append a dated entry after each completed task/session with commit, commands actually run, outcomes, benchmark report paths and precise next step. Keep incomplete/blocked/missed gates unchecked.
