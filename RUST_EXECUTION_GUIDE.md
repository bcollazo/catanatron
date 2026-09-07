# Execute the Rust rollout-engine plan

Read this first, then consult [the architecture and benchmark evidence](RUST_IMPLEMENTATION_PLAN.md). Update [the progress checklist](RUST_IMPLEMENTATION_CHECKLIST.md) after each task. The planning branch contains experiments, **no production Rust engine**. Everything under `rust/` below is a file or command to implement, not a claim that it already exists.

## Working instructions and scope

Work on the local branch provided by the user. Do not push, open a PR, merge unrelated work, replace Python's engine, or delete existing artifacts. Implement tasks E00–E13 in order, committing coherent tested increments if execution is authorized. A task is complete only when its exit checks pass; never check a box for a scaffold, stub, ignored test or hardcoded fixture response. If context runs out, record the next command and failing case in the checklist and resume from there.

The first release must provide: a complete BASE/TOURNAMENT Rust simulator for 2–4 players; legal generation separated from application; reproducible local rollouts; explicit chance; a v1 stdio bot; a basic useful rollout search; batched Python access suitable for later RL; and measured correctness/performance. MINI is included in E12 before final release. Training an RL policy, a fair hidden-state belief model, neural inference integration, GPU execution, sophisticated MCTS/TT/undo, and proving a world record are **follow-up work**, not reasons to delay a functional engine indefinitely. An in-process speed target can remain unmet even after functional tasks pass; report that honestly and keep the performance checklist open.

Use these defaults without re-litigating the architecture: safe Rust; edition 2021; fixed arrays plus `u64`/`u128` masks; ordinary typed action enum; reusable `Vec<Action>` scratch; copy-and-apply; exact bitset longest-road DFS; tile-local resource payouts; no production table, undo log, transposition table, SIMD or custom packed action encoding initially. Keep the experimental Cargo package separate from the production workspace.

## E00 — Establish the checkout, tools and baseline

1. Read root/nested `AGENTS.md` if present. Run `git status --short`, `git rev-parse HEAD`, `rustc -vV`, `cargo --version`, and a working Python's `--version`. Preserve user edits.
2. Baseline rules source is commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`; protocol/schema source is PR #386 head `5149b1869ba6318a2f2e3ef3925915576a433286`. Check whether current main has merged a newer #386. Record what is actually used. Do not silently mix newer rules with old expected outputs.
3. Use a virtual environment for Python. Install the core package and needed tests; avoid requiring web/database/Gym dependencies for core-only tests. Minimal probe dependency is `networkx==3.5`. On the planning host, `python` was an unconfigured pyenv shim; the working executable was `C:\Users\bcoll\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`. Discover an appropriate interpreter on other hosts rather than hardcoding this path into source.
4. Run the existing Rust probe with `cargo run --release --offline --manifest-path experiments/rust-rollout/Cargo.toml`. Consult its README to run Python/wire probes. Keep the recorded planning results; write new results under `rust/bench-results/<run-id>/` once created.
5. Create `rust/docs/provenance.md` recording revisions, tool versions, OS/CPU, and which actual commands ran. Do not describe unavailable checks as passed.

**Exit:** a working compiler/interpreter, baseline revisions recorded, probe assertions pass, existing Python core imports work. If a dependency/network/PR checkout is unavailable, record the exact external blocker, keep independent core work moving, and retain the affected integration gate as pending.

## E01 — Create the workspace and file ownership boundaries

Create this layout incrementally. Do not write all modules as empty placeholders and then call the architecture implemented.

```text
rust/
  Cargo.toml                         # workspace, resolver=2, release profile
  Cargo.lock                         # commit resolved dependency versions
  README.md                          # runnable examples and local validation commands
  crates/
    core/                            # package catanatron-core; std-only initially
      Cargo.toml
      src/{lib,ids,action,phase,position,topology,layout,rules}.rs
      src/{generate,apply,chance,awards,validate}.rs
      src/generated/{base,mini}.rs    # generated immutable geometry; never hand-edit
      tests/{topology,setup,builds,turns,chance,devcards,trades,awards,invariants}.rs
    search/                          # catanatron-search: RNG, policies, rollouts, workers
      src/{lib,rng,policy,rollout,flat_search,batch}.rs
    bot/                             # binary catanatron-bot, JSON only at boundary
      src/{main,wire,import,export}.rs
      tests/{protocol,snapshots}.rs
    bench/                           # binary catanatron-bench, conformance binary
      src/main.rs
      src/bin/catanatron-conformance.rs
    python/                          # E12: optional catanatron-python extension
  tools/{export_topology,export_fixtures,differential,compare_bench,verify_stdio}.py
  tests/fixtures/{manifest.json,topology,transitions,protocol,divergences}/
  docs/{rules-profile,fixture-format,provenance,performance}.md
  bench-results/                     # ignore bulky runs; commit selected reports
```

Use package names above so subsequent commands stay valid. Set the bench package's `default-run = "catanatron-bench"` because it also contains the conformance binary. Set workspace `default-members` to core/search/bot/bench once present; keep the Python extension optional so basic Rust checks do not need Python linker configuration. Add crates as they become useful, not nonexistent workspace members. Release profile: `opt-level=3`, `lto="thin"`, `codegen-units=1`; keep debug assertions/overflow checks in debug tests. Keep native CPU flags out of distributable default builds.

Dependencies: core starts std-only; serde/serde_json belong to wire/fixture executables, not `Position`; use `rand_chacha` with `ChaCha8Rng` in search as a conservative initial RNG, with its compatible RNG trait dependency; PyO3/NumPy/maturin only in E12. Resolve compatible versions against their official docs when implementing, commit `Cargo.lock`, and record RNG test vectors. Avoid adding a web server, async runtime or graph crate. Export a tiny core random-source trait or pass deterministic outcomes so core does not depend on search.

**Exit:** `cargo check --manifest-path rust/Cargo.toml` and `cargo fmt --check --manifest-path rust/Cargo.toml` pass; core depends on neither Python nor JSON nor search. README explains that performance probes are a separate package.

## E02 — Freeze rules and produce complete fixtures

Read Python `models/actions.py`, `apply_action.py`, `state_functions.py`, `models/board.py`, `models/decks.py`, and matching tests before porting a subsystem. The original architecture plan leaves rule review open; use this explicit release policy to resolve it:

| Topic | Initial release policy |
|---|---|
| Inventories/costs | Resource order WOOD, BRICK, SHEEP, WHEAT, ORE; bank 19 each; pieces 15 roads/5 settlements/4 cities per player; development counts KNIGHT=14, YEAR_OF_PLENTY=2, MONOPOLY=2, ROAD_BUILDING=2, VICTORY_POINT=5. Costs: road `[1,1,0,0,0]`; settlement `[1,1,1,1,0]`; city `[0,0,0,2,3]`; dev `[0,0,1,1,1]`. |
| Bank shortages | Match pinned `yield_resources`: if aggregate demand for a resource exceeds its bank count, pay none of that resource to anyone on that roll. Other resources still pay. |
| Maritime menu | Match pinned generator: only the best available 4:1/3:1/2:1 rate per given resource. No identical-resource trade. Keep a documented extension point for a later all-rates profile. |
| Year of Plenty | Port the exact candidate algorithm, including singles added when a candidate pair is unavailable, then canonicalize/deduplicate. Do not replace it with “singles only when total bank cards=1”; that differs from this checkout. |
| Development eligibility | Store counts plus a four-type start-of-turn eligibility mask and played-this-turn flag, matching Python. Newly bought VP counts immediately. A non-VP card bought this turn cannot create eligibility. At most one non-VP card is played per turn. |
| Victory | At each completed player action (including its chance resolution), reproduce Python's winner scan over active seats; last qualifying seat wins if a constructed fixture has multiple qualifiers. Stop before generating further moves. Empty terminal menu is an intentional API normalization, not a Python menu mismatch. Do not replace this with current-turn-only victory during the initial port. |
| Turns/cutoff | Preserve Python `num_turns`, including setup's `advance_turn` calls. Default turn limit 1,000. Winner has precedence over cutoff. Distinguish a separate search action-budget cutoff from game termination. |
| Friendly robber | Match Python: exclude candidate tiles touching an opponent with actual VP <3; if this removes all robber actions, fall back to the unfiltered menu. Default off. |
| Longest road | Use the exact edge-simple-trail definition in the architecture plan, including incoming roads at opponent buildings. Use correct incumbent tie/removal behavior specified in E06. Record targeted discrepancies with Python as named divergences; do not port broken connected-component bookkeeping. |
| Domestic trading | Ask each other seat once in ascending seat order, skipping the proposer. Require nonnegative bounded counts, both sides nonempty, no resource on both sides, and enough resources on confirmation. The pinned response-advance code may visit the proposer; write a targeted repro instead of copying it. Protocol v1 still cannot originate an offer. |

Name the documented engine profile `rust-v1`, derived from the pinned Python profile with only the listed normalizations/corrections. This is not a claim of universal official-rule conformance. For longest-road/trade discrepancies, create minimized Python input/output and explicit Rust expected-output fixtures in `divergences/`. Keep a manifest of the exact differing fields and rationale. **Never suppress an entire failed game or broad action type.** A newly discovered discrepancy requires a minimized repro, a deliberate documented decision and a regression test before it can be accepted.

Write `export_topology.py`: generate Python BASE and MINI geometry, sort undirected land edges by endpoint pair, emit vertices/edges/land tiles/coordinates/ports/active masks and generated Rust tables. Validate BASE counts 54/72/19, no duplicate undirected edge, reverse lookup bijections, edge endpoint incidence, node degree <=3 and six vertices per land tile. Keep water/port tile IDs distinct from dense land-tile indices; preserve Python node/coordinate mappings. Do not assume MINI is a prefix of BASE without checking the exported mapping.

Write `export_fixtures.py`: capture initialized snapshots and forced action/outcome transitions, plus sampled complete-game traces. Each JSONL record has `fixture_version`, `case_id`, `source_revision`, `rules_profile`, `before`, `actor`, **intent** `action`, optional `outcome`, `after`, `legal_before`, `legal_after`, and `status_after`. The manifest records resource/dev orders, map tables, seeds, per-file SHA-256, covered actions/phases/configurations and explicit divergence IDs. Store wire triples alongside canonical typed representations so both converters are tested. Prefer small committed golden fixtures plus deterministic generators for large corpora.

Canonical `before/after` fields: map/rules ID; active seats and wire colors; ownership arrays (building owner plus settlement/city kind, one owner per road); bank and each hand/dev count; eligibility/played flags; piece inventories; award holders/lengths; robber; actor/turn owner; typed phase payload; pending trade/responses; Python-compatible turn count. Exclude caches, UUID, action history and RNG internals. Represent chance-pending states only in Rust-only tests; Python records compare boundaries after an entire intent+outcome has completed.

**Critical replay trap:** Python records replace `ROLL.action.value` with the dice pair and `BUY_DEVELOPMENT_CARD.action.value` with the drawn card. Recover intent with `value=None` for these two types, and carry the result separately. `DISCARD_RESOURCE` is deterministic although its record repeats the discarded resource. A `MOVE_ROBBER` without a victim has no theft outcome. Preserve tuple/list and color conversions at the boundary; do not compare raw JSON object ordering.

**Exit:** fixtures cover all 18 action variants and all reachable decision phases (crafted trades/devcards included), 2/3/4 seats, BASE/TOURNAMENT, pre/post-roll resume, and the named divergences. Committed topology can build offline. Exporting twice from the same inputs gives identical canonical fixtures/hashes.

## E03 — Implement types, state and checked boundaries

Use private `u8` newtypes for IDs with checked constructors and dense resource/dev enums. Explicitly map resource/dev orders; Python's deck construction order is not an enum ABI. Building ownership/kind can be a small value enum or byte encoding with safe accessors. Use only active seats when iterating `[PlayerState;4]`; zero unused entries deterministically.

`Position` holds exactly the canonical mutable fields above plus measured caches. Keep config/topology/layout in a borrowed `GameContext`; copied positions do not own `Arc`s, vectors, logs, RNGs or strings. Store `dev_counts[5]`, `eligible_dev_mask`, and `played_dev_this_turn`. Do not invent old/new counts from v1 snapshots: they expose eligibility booleans, not precise age counts, and the one-card limit makes the mask sufficient.

Define `Action` as in the architecture plan, with `Copy/Eq`; `Phase` with its required payload; `Outcome` (`Dice`, `StolenResource`, `DevelopmentCard`); `Status` (`Decision`, `Chance`, `Won(player)`, `Truncated(reason)`). Store pending chance in `Phase`, including resume state, so it survives copying/hashing and forbids a second action before resolution. Outcomes are not player actions. Deterministic transitions need no fake outcome.

Public checked calls return typed errors for wrong actor/phase/ID, insufficient resources, invalid geometry, exhausted inventory, incompatible outcome and terminal state. Failure leaves canonical state and caches unchanged. The crate-private apply-generated function avoids redundant whole-menu membership checks but uses the same transition helpers. Core operations never print or invoke observers.

**Exit:** size/layout tests record `Position`/`Action` sizes; invalid-ID, phase and failure-atomicity tests pass; copying is independent; no core heap-owning field; dormant flags cannot contradict the typed phase. A position larger than 512 B must be explained, not squeezed with unsafe packing.

## E04 — Setup, construction and ordinary turns

Implement generation and application together for each row below. Every generated action must pass checked validation; applying it through checked/generated paths must yield equal state/status. Payment moves resources back to bank; free placements do not. City returns one settlement piece, consumes a city and adds one net public VP.

| Current phase | Menu / transition |
|---|---|
| Setup settlement | Any distance-legal empty land vertex; grant second-settlement adjacent non-desert resources; remember this node; same actor -> setup road. |
| Setup road | Empty edge incident to that remembered node. Seat order for four players is `0,1,2,3,3,2,1,0`, each making a settlement+road pair. After the final road, seat 0 is pre-roll. Generalize to N seats. Match setup turn-counter changes. |
| Pre-roll | Roll plus eligible dev-card plays; no paid builds/trades/end turn. Roll requests a dice outcome. |
| Post-roll | EndTurn, affordable/legal roads/settlements/cities/dev purchase, best-rate maritime trades and eligible dev-card plays. No second roll. |
| EndTurn | Clear outgoing player's turn flags; refresh its dev eligibility from held cards as Python does; advance actor and turn owner modulo active players and increment turn count; enter pre-roll. |
| Terminal/cutoff | Empty menu; reject further actions until an explicit reset/new session. |

Road predicate: an edge is empty and one endpoint is an own building OR an empty vertex incident to an own road. Enemy buildings block extension through that vertex. Settlement predicate: unoccupied and no neighboring building; outside setup, incident to own road. City predicate: own settlement. Check inventories and affordability before geometry enumeration when possible. Use deterministic ID order, without assuming it is Python's list order.

Implement `generate_actions(ctx, &position, &mut Vec<Action>)` with an explicit clear-at-entry contract. Initially reserve 256 entries once per worker for ordinary generated menus; keep safe `Vec` growth, instrument capacity, and prove/check an upper bound before claiming zero allocations across every supported mode. Domestic proposals are not exhaustively enumerated.

**Exit:** topology/setup/build tests pass for each seat count, including blocked endpoints, coast/water rejection, distance, ports, depleted pieces, returned settlement piece and second-settlement payouts. No turn-loop placeholders.

## E05 — Dice, production, discards and robber

`Roll` marks the roll consumed and enters pending dice. Resolving sum !=7 builds aggregate resource demands using matching tiles and current buildings, omits robber tile, applies per-resource shortage policy, then returns post-roll. Sum 7 sets each discard amount once (`hand_total/2` iff total > configured limit), starting with the lowest seat needing discard. Each `Discard(resource)` returns exactly one card to bank; continue that seat until done, then next higher qualifying seat; finally restore actor to turn owner and enter robber selection. Do not reroll while discarding.

Robber menu: every other land tile; for each tile, one action per distinct opponent owner with >=1 resource card, or a single no-victim action if none qualify. Apply friendly-robber filtering/fallback when configured. For a victim, request a theft outcome weighted by the victim's resource counts; validate the resource is present before transfer. No-victim movement is deterministic. Resume pre-roll for a pre-roll knight and post-roll for a seven or post-roll knight.

Chance enumeration returns integer `(outcome, weight)` entries and a denominator; probabilities sum exactly. Dice may group sums but replay accepts concrete dice pairs. Invalid impossible draw/theft/dice results must not partially mutate state. For authoritative deck fixtures, preserve ordered deck in session storage and force its next result; sampled rollout positions use counts.

**Exit:** tests for all 36 dice pairs/sum weights, per-resource shortages, city double yield, robber suppression, multiple discarders out of turn, victim deduplication, empty victims, weighted theft and parent independence. Sampled frequencies are a sanity check; exact weights are the correctness oracle.

## E06 — Development cards, awards and domestic trade

Implement all actions, with fixtures for unavailable cards and newly bought cards. BuyDev pays once, requests a possible draw, decrements deck count, increments hand and immediate hidden VP if applicable. Eligibility refresh must not make a just-bought non-VP card playable this turn. Knight consumes the turn's dev allowance, increments played knights/army and enters robber with correct resume phase. Plenty and Monopoly transfer resources atomically; RoadBuilding consumes the allowance and enters a free-road sequence (requires at least one available legal road to be offered). After each free road, decrement remaining count and finish early if no road/piece remains. Preserve pre/post-roll resume.

Implement award recomputation separately from geometry. Longest-road length counts edges with no edge reused; revisiting a vertex is legal; opponent buildings stop continuation after an incoming edge. Minimum award length 5. If the incumbent ties for the maximum >=5, retain it; otherwise award a unique maximum >=5 or leave unheld on a tie/no qualifier. Correctly remove the old +2 VP when unheld or transferred. Largest army minimum 3, incumbent retains ties, transfer only to a strictly larger eligible army. Public VP = settlements + 2*cities + awards; actual VP additionally includes VP cards. Use derived/recomputed values in tests to detect double counting.

Trade flow: offer in post-roll -> other seats respond in ascending order -> proposer chooses an accepter or cancels -> post-roll. Reject always available; accept only if responder can pay the requested bundle; no resource moves on accept. Confirmation validates both holdings again, exchanges bundles and clears pending trade/response state. With no accepters, return directly to post-roll. Enforce bounds and prevent underflow even where Python's checked path is permissive. Do not advertise the ability to send offers through v1 stdio.

**Exit:** no unsupported action variant remains; cards/awards/trades tests and action/outcome fixture matrix pass. Include road loops/tails, blocked degree-3 junctions, award loss/ties, every proposer seat in 2/3/4-player trade sequences and failed-confirmation atomicity. Named Python divergences have narrow expected results, not skipped tests.

## E07 — Complete games, random streams and rollout API

Implement map initialization and seat assignment explicitly. First use imported fixed layouts; then implement random resource/port assignment and supported `official_spiral`/`random` number placement from Python templates. Shuffle with unbiased Fisher–Yates. Do not promise equal boards for equal Python/Rust seeds; use exported layouts for exact comparisons. TOURNAMENT has fixed assignment; respect valid number-placement combinations from Python's CLI. Random layouts must satisfy the chosen placement algorithm's constraints, not extra invented restrictions.

Core RNG interface provides `next_u64`; a bounded draw uses rejection, never bare `% bound`. Search's initial implementation wraps a pinned `ChaCha8Rng`. Derive a stable 32-byte child seed from master seed, game index, rollout ID and stream kind using a documented fixed integer-mixing procedure; add known-answer vectors. Do not use Rust `DefaultHasher`, wall time, thread ID or unspecified hashes as seed derivation. Separate policy choices from chance draws. Per-rollout stream identity must be independent of worker scheduling.

Policies: uniform over actual menu entries; weighted policy gives each city action weight 10000, each settlement 1000, each dev purchase 100 and every other action 1 (as Python). Select by cumulative integer weight instead of physically duplicating entries. Do not prune legal actions in these baseline policies.

Expose `rollout(ctx, root, policy, seed, limits, scratch) -> RolloutResult` with winner/truncation, turns and player-action count. Copy root once, then apply/sample/resolve/generate forward. Count one player intent as one action transition even if it has an internal chance phase; count each discard/free-road decision separately. Maintain a separate total-action cap to stop infinite maritime cycles in search; label cap results truncated and report them, never fabricate a winner. Full-game comparators use identical turn/action caps.

**Exit:** complete 100 seeded games for 2 and 4 players, random and weighted, plus deterministic fixture layouts; same Rust inputs reproduce identical traces; scalar rollouts do not change roots; every nonterminal decision has a supported menu or explicit parameterized-proposal policy. Resource/development/piece conservation assertions pass throughout debug trajectories.

## E08 — Differential harness and performance scoreboard

Implement the `catanatron-conformance` JSONL executable. It imports canonical before states, validates/generates the menu, applies intent+forced outcome, exports canonical after/status/menu, and exits nonzero with the first detailed mismatch. Keep serde DTOs in this executable/bot helper, not the core. `differential.py` drives both engines with the same selected intent/outcome and reports the first differing field. It must fail unexpected divergences. After an accepted terminal transition, compare normalized terminal menus; do not execute another action.

Required corpus: every golden case from E02, at least 100 full controlled-outcome trajectories per 2/3/4-player configuration, and crafted max-inventory/rare-phase graphs. A registered divergence can be checked as its own standalone fixture; if it is reached in a paired trajectory, report that trajectory separately rather than counting it as a fully equal game. Publish equal, divergent, truncated and failed counts. Use independent complete Rust games for coverage beyond divergences.

Implement `catanatron-bench` with subcommands `games`, `rollouts`, `kernels`, `allocations`. Common args: `--seed`, `--players`, `--map`, `--policy`, `--threads`, `--output`; games adds `--games`, rollouts adds `--fixtures` and `--rollouts`. Parse invalid CLI options with errors. Use a fixed tiny dependency or straightforward parser; do not build CLI infrastructure for its own sake.

Results JSON includes revision, profile/map/policy, seeds/fixtures, CPU/compiler/build flags/threads, sample durations, intent/chance/turn/game counts, completed/truncated/divergent counts, throughput and state size. Warm up, run >=5 batches, retain raw samples. Compare identical workloads on the same host; do not compare Rust's weighted policy to Python's random policy or combine compatibility-divergent trajectories into an “exact” speed ratio. Add allocation counting around warmed *rollout execution only*, excluding input loading/results printing/thread setup; include worst-case menus and all phases.

**Exit:** `differential.py` passes all expected cases with zero unexplained differences; full-game and fixed-state rollout reports exist; allocation count is zero for the supported baseline path after scratch setup. Target >=10x same-machine Python throughput and >=1M player intents/s one core. If missed, identify top costs, run one controlled optimization at a time, and remeasure. A numerical target missed is recorded as missed, not rationalized away or satisfied using a different workload.

## E09 — Stdio adapter and snapshot importer

Read the pinned PR files (or merged successor after explicitly recording the change). Build the exact v1 messages in the architecture plan. Default `observe=false`. Buffer stdin by lines; cache map on `before`; on every `decide`, refresh dynamic state and clear stale search state. Verify protocol/schema versions, game ID, active colors, duplicated road consistency and every numeric bound. Diagnostics go to stderr; replies are one newline-terminated flushed JSON object, only for hello/decide. Exit cleanly on EOF. Unknown notification types may be ignored; malformed decisions produce a clear failure rather than invented actions.

Import uses `colors` as seat order, never the enum's declaration order. Read `P<i>_*` fields into corresponding seats; preserve eligibility flags and incumbent awards. Map Python `current_prompt` first, then relevant free-road flags and `HAS_ROLLED` to choose a resume phase. Some auxiliary booleans are stale in the pinned implementation (for example robber flags); do not reject a valid snapshot solely because an unused legacy flag disagrees with the prompt. Setup's latest settlement comes from ordered `buildings_by_color`, not node ID order. `current_trade` has its own seat-index payload; action wire colors remain color names.

Handle authoritative ordered dev deck vs bot count object with separate parsers. The bot snapshot contains opponent holdings: mark imported search mode `PerfectInformationV1`. Avoid processing full history for ordinary import; JSON can deserialize known fields while ignoring `action_records`, but measure parsing cost. The `step` notification does not contain random outcomes; never rely on it for root advancement.

First bot policy: choose a supplied action at random and echo its **original wire triple**. Then enable Rust menu comparison and rollout search. Map root actions by typed semantic value. Any known rule divergence gets an explicit diagnostic/counter and fixture; do not silently map to “close enough” actions or claim menu parity. Default on unexplained root mismatch is a clear integration error during validation. In release play, any configured emergency choice must be an offered legal action and counted separately as degraded behavior.

Implement `verify_stdio.py` using the real PR adapter in a separate environment/worktree if the current checkout lacks it. Do not transplant half the PR into main. Also keep a hermetic fake-host test suite so engine/bot development works before merge. Test multiple games in one process, skipped forced decisions, lost optional notifications, wrong IDs, EOF, invalid fields, all wire action payloads and exact reply count. PR v1 lacks request IDs: bot never pipelines, never emits a stale late reply, and uses a configured time budget shorter than host timeout.

**Exit:** fake-host tests and at least 100 mixed Python/Rust games pass with zero unexpected mismatches, timeout, illegal-action or random-forfeit incidents; real-host counters included. If the required host revision is externally unavailable, leave real-host certification pending and finish independent work. Root import must not be treated as a complete oracle for future hidden-state inference.

## E10 — A useful search bot with bounded work

Implement flat Monte Carlo first, not a full MCTS tree. For each supplied root action, run independent simulations that apply it, sample any immediate chance, then use the weighted rollout policy to terminal/cutoff. Maintain per-root-action sum/count; terminal reward for the root player is 1 for win, 0 for loss, 0.5 for cutoff (document this search convention; it is not the RL reward). Choose highest mean, deterministic action-key tie break. Allocate samples round-robin so early action ordering cannot consume the whole budget. At least one valid offered action is always retained before search starts.

Expose bot CLI options `--policy random|rollout`, `--simulations`, `--budget-ms`, `--seed`, `--threads`. Default budget 100 ms under the host's default 1000 ms timeout, starting when a decide line is received. Deadline checks occur between rollouts and at bounded action intervals inside them; expensive DFS must also have a bounded inventory, and measure tail latency. Stop with enough time to serialize/flush. A one-action menu returns immediately. Fixed-simulation single-thread mode provides deterministic tests; deadline mode's number of completed rollouts is not guaranteed deterministic.

**Exit:** forced-win/no-loss synthetic decisions behave correctly, all replies are legal, same fixed simulation seed gives same result, and parent root never mutates. Report rollouts/decision, decision p50/p95/p99 and head-to-head results versus R/W over a fixed seed/seat-rotation schedule. Report confidence/sample size; a small win-rate observation is not evidence of strength superiority. MCTS/TT are later optimizations after this usable baseline.

## E11 — Reproducible parallel batch execution

Implement search's Rust `Batch` and parallel `rollout_many` with fixed worker chunks or a pool outside the core. Each worker owns position/scratch/RNG, writes results to deterministic input indices, and aggregates once per batch. No shared RNG or per-action mutex. Use independent `(game, rollout)` seeds from E07. Root/decision deadlines remain owned by the adapter, not embedded in `Position`.

**Exit:** for 1/2/4 threads, fixed input+simulation count returns exactly the scalar per-rollout results regardless of scheduling; disjoint games cannot corrupt each other. Publish throughput at 1/2/4/8/16 physical cores and SMT where available, including small and large batches and memory. Missing hardware sizes are recorded as unavailable. Do not claim linear scaling from CPU count alone.

## E12 — MINI and batched Python access for later RL

Complete MINI through exported topology/active masks, bank/map-specific initialization and documented dice behavior from the pinned Python implementation; no invented rerolling of sums absent from the board. Rerun the action/phase/differential tests and scalar games. TOURNAMENT uses BASE topology and fixed assignment; import/export must preserve that assignment.

Add the optional extension package and a separate Python build `pyproject.toml` under `rust/crates/python/`, using maturin/PyO3 and a compatible NumPy bridge. Keep the repository's existing Python package metadata intact. Provide an extension module such as `catanatron_rust` and `Batch` methods:

| Method | Contract |
|---|---|
| `reset_many(indices, seeds, config)` | Explicitly reset selected environments; independent seeds; no implicit reset on terminal. Return/update observations and menus for selected environments. |
| `step_many(indices, action_ids)` | One acting-player intent per selected environment, resolving its chance internally. Return actors, rewards, terminal/truncated flags and next legal masks/menus in contiguous arrays. A different method is needed to auto-play opponents until a fixed player's next decision. |
| `observe_many(indices)` | Pure batch observation/legality query; no mutation/RNG draws. Include observation/action schema versions. |
| `rollout_many(indices, seeds, limits, policy)` | Copy each selected root and simulate; leave batch roots unchanged; return compact counts/results. |

For the initial self-play API, terminal rewards are a vector over active players: +1 winner, -1 losers, otherwise 0, with truncation separately flagged. Never reuse flat-search cutoff reward 0.5 here. Use perfect-information observations labeled as such. Fair observation/belief sampling is outside this release. Version a flat numeric feature order and store its specification; do not claim compatibility with Python's `create_sample` vector unless every feature is actually ported and compared.

Provide an explicit adapter for existing Gym **action catalogue indices** by exporting the Python catalogue for each map/player-color configuration and mapping semantic actions bijectively. It excludes domestic offers, just as the existing catalogue does. For generalized trade-capable self-play expose a separate versioned dynamic menu with offsets and opaque `u64` action IDs encoding a decision-generation counter and row index; never present dynamic IDs as stable Gym IDs. Increment the generation on reset and every state transition, reject counter exhaustion explicitly, and reject stale/out-of-range dynamic IDs without mutation. Stable Gym catalogue IDs only require current-menu legality; they do not carry a staleness guarantee. Terminal masks are empty.

Define batch error policy before code: validate every supplied index/action against current menus before any environment mutates, reject duplicate indices in a batch, then execute. Return numeric arrays with documented shapes, dtypes and ownership lifetimes; do not leave NumPy views pointing into scratch that is freed/reallocated. Release the GIL during pure Rust execution; no Python callbacks inside the rollout loop. Add a short example with reset/step/masks/termination and a timing script; do not start RL training.

**Exit:** extension builds in a fresh environment; batch-size 1 matches scalar Rust; fixed seeds match for batches 1/16/256 and multiple worker counts; invalid-batch calls are atomic; no stale views; masks/actors/rewards correct through discards/trades/terminal/reset; mapped Gym action values agree; samples/s include observation and reset costs. Benchmark Python crossing per batch rather than per simulated action.

## E13 — Final validation, documentation and handoff

Create and run the following commands after implementing their CLI contracts. Commands are from repository root; replace Python with your environment interpreter and add `.exe` to binary paths on Windows. Dependency-backed commands need an initial online install/build; use offline mode only after caching dependencies.

```text
cargo fmt --check --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
cargo test --release --manifest-path rust/Cargo.toml -p catanatron-core -p catanatron-search
python rust/tools/export_topology.py --check
python rust/tools/export_fixtures.py --check
python rust/tools/differential.py --fixtures rust/tests/fixtures --games-per-config 100
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- games --games 1000 --players 4 --map BASE --policy random --seed 0 --threads 1 --output rust/bench-results/final-games.json
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- rollouts --fixtures rust/tests/fixtures --rollouts 10000 --players 4 --map BASE --policy random --seed 0 --threads 1 --output rust/bench-results/final-rollouts.json
cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench -- allocations --seed 0 --players 4 --map BASE --policy random --threads 1 --output rust/bench-results/final-allocations.json
cargo build --release --manifest-path rust/Cargo.toml -p catanatron-bot
python rust/tools/verify_stdio.py --bot rust/target/release/catanatron-bot --games 100
python -m maturin develop --release --manifest-path rust/crates/python/Cargo.toml
python -m pytest rust/crates/python/tests
```

`--check` means regenerate in temporary storage and compare, without overwriting goldens. Build/test the optional extension using its documented environment separately from default workspace checks. Add a focused Rust CI job for fmt/clippy/test/fixtures and extension smoke tests; do not make CI speed judgments on unstable shared runners. Existing Python core tests must still pass if any shared Python tooling changed; no need to run unrelated PostgreSQL/UI infrastructure when untouched.

`rust/README.md` must contain installation, a complete in-process example, actual stdio invocation with correct `rust/target/release` path, configured budgets, a batch Python example, supported profiles/maps/config limits, known exact rule divergences and fair-information limitations. `rust/docs/performance.md` contains reproducible before/after data and which speed targets were achieved or missed. Add an implementation overview link from the root README when the implementation is usable.

Final handoff reports functional checklist status, actual revisions, checks run, benchmark changes, known failures/divergences and next optimization. **Functional readiness, protocol certification, performance targets and “fastest” evidence are four separate statuses.** Never mark an external integration check or numerical target complete because time/context is low. No remote publication is part of this task.

## Resume procedure and troubleshooting

At the end of each execution session update the checklist with current task, last passing command, exact failing command/output summary, changed files, fixture IDs and next concrete action. Read it before resuming. Run targeted checks for the subsystem just changed; broaden to full checks at the stated exits.

* Wrong action menu: compare actor/phase first, then dev eligibility/costs, then geometric masks; print canonical differing action values, not only a count.
* Same move, different state: force the same chance outcome; compare resource transfers, counters/resume phase and awards before inspecting caches.
* Diverges only after copying: inspect accidental shared scratch/RNG/history and forgotten phase payloads.
* Diverges only after import: inspect seat order, duplicate road orientations, dev-count object vs ordered list, stale legacy flags and setup settlement ordering.
* Fast kernels, slow full game: profile generation frequency, allocations, RNG/policy overhead, longest-road recomputation, map initialization and feature work. Never add protocol into rollout timing.
* No legal moves: distinguish terminal, pending chance, free roads exhausted, empty discard count and incomplete trade queue before adding a fallback. A random action cannot repair an invalid phase.
