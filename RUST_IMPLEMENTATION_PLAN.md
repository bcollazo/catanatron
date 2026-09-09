# A Rust Catan engine for search and RL rollouts

Prepared 2026-09-07. Python baseline: `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.
Reviewed [PR #386 — Rework custom bots: shared registry, one lifecycle, bots in any language](https://github.com/bcollazo/catanatron/pull/386), open at review time, specifically head `5149b1869ba6318a2f2e3ef3925915576a433286`.

**Implementer entry point:** follow [RUST_EXECUTION_GUIDE.md](RUST_EXECUTION_GUIDE.md) in task order and maintain [RUST_IMPLEMENTATION_CHECKLIST.md](RUST_IMPLEMENTATION_CHECKLIST.md). This document gives architectural reasoning and measured evidence; the execution guide supplies concrete defaults, contracts, commands and completion gates. Suggested later optimizations are not prerequisites for the first complete engine.

**Recommendation:** build an independent, safe-Rust simulation library with compact fixed-size state, immutable board tables, typed atomic actions, separate move generation and mutation, explicit chance outcomes, and no allocation or history logging in the rollout loop. Start with copy-and-apply. Connect a Rust bot through PR #386's stdio adapter; keep every search simulation inside that Rust process. Add batched Python bindings for RL after the engine is correct and measured.

I interpret the request's architecture sentence as “it does **not** have to follow Python.” Preserve the rules and integration contracts, not Python's object layout. “Fastest Catan implementation ever” is the optimization objective; the experiments here establish promising directions, **not** a completed engine or a world-speed record.

## 1. What this repository contains

| Area | Current responsibilities | Rust implication |
|---|---|---|
| [`game.py`](catanatron/catanatron/game.py) | Game loop, bot dispatch, validation, winner checks, copy, and accumulators. `execute()` calls application and then always generates another action list. | Keep orchestration outside the kernel; callers choose whether to generate moves. |
| [`state.py`](catanatron/catanatron/state.py), [`state_functions.py`](catanatron/catanatron/state_functions.py) | Mutable state and helpers. Player data uses keys such as `P0_WOOD_IN_HAND`; state contains phase flags, trade state, player objects, decks and history. | Typed fields, indexed players/resources, one explicit phase representation. |
| [`models/actions.py`](catanatron/catanatron/models/actions.py), [`apply_action.py`](catanatron/catanatron/apply_action.py) | Already separate generation and mutation. Application dispatches on action type and records stochastic results. Helpers sometimes regenerate possibilities or revalidate board placement. | Preserve separation and consolidate legality so trusted rollout moves avoid duplicate work. |
| [`models/enums.py`](catanatron/catanatron/models/enums.py) | `Action(color, action_type, value)` and `ActionRecord(action, result)` namedtuples; payload shape depends on the tag. | Rust sum types remove unrelated payload combinations. |
| [`models/board.py`](catanatron/catanatron/models/board.py), [`models/map.py`](catanatron/catanatron/models/map.py) | NetworkX topology, dictionary ownership, both orientations of every road, connected-component sets, buildability/port caches and longest-road search. BASE has 54 land vertices, 72 edges, 19 land tiles. MINI and fixed-assignment TOURNAMENT maps also exist. | Dense IDs and fixed adjacency tables; no general-purpose graph library in the hot path. |
| [`players/`](catanatron/catanatron/players) | Random/weighted/value policies, minimax/alpha-beta, MCTS, chance expansion and multiprocessing playouts. Search copies games and expands outcomes. | An engine/search boundary that supports sampling, enumeration and cheap state forks. |
| [`features.py`](catanatron/catanatron/features.py), [`gym/`](catanatron/catanatron/gym) | Feature extraction, Gymnasium environment, observations and a sorted catalogue mapping actions to RL indices. Current mapping uses linear list lookup. | Versioned explicit action codecs, direct indexing and batched features; do not silently invalidate trained policies. |
| [`cli/`](catanatron/catanatron/cli), [`web/`](catanatron/catanatron/web), [`ui/`](ui) | Simulation CLI, dataset accumulators, Flask/persistence backend and React UI. | Retain these as clients; they need not be rewritten to get fast rollouts. |
| [`tests/`](tests), [`catanatron_experimental/`](catanatron_experimental), [`documentation/`](documentation) | Rule/regression tests, integration tests, experiments, training examples and docs. Existing speed tests include mutable game benchmarks and setup-state copy tests. | Reuse rule scenarios, but add fixed-corpus performance and differential tests. |

Python already has useful performance ideas: frequency decks, immutable map sharing, cached buildability, and custom copying. The rewrite should carry these ideas forward without carrying string-keyed dictionaries, duplicated road entries, graph objects, or pickle/deepcopy operations. `State.copy()` copies history, and `Board.copy()` serializes connected components and deep-copies caches; search cost consequently depends on more than the rule state.

The project is GPL-3.0-or-later. Keep the new implementation in this repository under the existing project license.

## 2. What PR #386 enables—and its exact boundary

Read the actual pinned [protocol implementation](https://github.com/bcollazo/catanatron/blob/5149b1869ba6318a2f2e3ef3925915576a433286/catanatron/catanatron/protocol.py), [stdio adapter](https://github.com/bcollazo/catanatron/blob/5149b1869ba6318a2f2e3ef3925915576a433286/catanatron/catanatron/players/stdio.py), [serializer](https://github.com/bcollazo/catanatron/blob/5149b1869ba6318a2f2e3ef3925915576a433286/catanatron/catanatron/serialization.py), and [example bot](https://github.com/bcollazo/catanatron/blob/5149b1869ba6318a2f2e3ef3925915576a433286/examples/stdio_bot.py), not just the PR description.

### Protocol v1 integration

| Message | Rust behavior |
|---|---|
| `hello` | Reply with `{"protocol_version":1,"name":"rust-rollout","observe":false}`. Version mismatch is fatal. |
| `before` | No reply. Cache `game_id`, color/seat mapping, map assignment and derived geometry. Reset per-game search state. The subprocess survives across games. |
| `decide` | Import the dynamic snapshot, attach the cached map, search locally, then return `{"action":[color,action_type,value]}` plus newline and flush. |
| `step` | Optional, no reply. Carries an action intent, **not** its random result. Useful for notification, insufficient by itself for exact state synchronization. |
| `after` | No reply. Release per-game search data and record winner/truncation as appropriate. |

Only `hello` and `decide` have replies. Send diagnostics to stderr. `decide.state` omits only `map`; it still includes action history. The adapter skips calling the subprocess when exactly one action is legal, so the bot cannot assume it sees every decision. The PR has no request IDs or advertised per-request time budget. Use a configured local deadline shorter than the host timeout (default 1,000 ms), and emit exactly one timely reply. Do not pipeline requests.

The host compares the reply to serialized entries in `playable_actions`; it does not accept a newly invented action just because its payload parses. Map each offered wire action to a typed internal action and retain the original wire value for the root response. This avoids encoding/order differences in road endpoints, Year of Plenty tuples, colors, and maritime-trade null padding. Rust move generation must independently agree with the offered menu in conformance tests.

Expected invocation after the Rust binary exists:

```text
catanatron-play --bot RUST=exec:"./rust/target/release/catanatron-bot" --players=R,R,R,RUST --num=100
```

Use `.exe` on Windows and pass a configured timeout when search needs one. This command is an implementation target, not something implemented by this plan.

### Important compatibility details

* The new registry, frozen parameter dataclasses, common `before/step/after` lifecycle and JSON persistence solve bot registration and process lifetime. They do not make the Python referee a Rust simulator. The first deliverable is a **Rust bot with a private Rust rollout engine**, with Python still refereeing the match.
* `client_view` removes the seed and replaces the ordered development deck with remaining counts. It **does not redact opponents' resource/development hands or history**. Treat v1 as the existing largely perfect-information comparison mode, with unknown future deck order. It is not a fair imperfect-information observation contract. Coordinate future fair play with the separate observation work; never train/evaluate a fair agent by silently giving it these hidden fields.
* `state_to_json` and `client_view` are different input schemas in practice: `development_listdeck` is a list in an authoritative save, a count object in a bot snapshot, despite retaining the same key. Use distinct import paths and validate `schema_version` separately from `protocol_version`.
* The serializer describes map assignment and a BASE/MINI topology name, not explicit geometry. Export and test Python's deterministic vertex/edge/coordinate mapping; derive geometry from those versioned tables. TOURNAMENT round-trips as BASE plus assignment. Collapse duplicated directed road entries and reject conflicting owners.
* Snapshot caches such as `connected_components`, road lengths and buildable IDs need not dictate Rust storage. Recompute/check them from canonical ownership during import. Compare awards separately because ties depend on the incumbent.
* Main's Python random stream is global, and this schema contains no complete RNG state. Exact differential testing must feed recorded chance outcomes, not expect Rust and Python to produce the same game from an integer seed.
* Python permits parameterized `OFFER_TRADE` outside its generated list; protocol v1 only accepts members of that list, which currently contains no new trade proposals. A Rust bot therefore cannot originate domestic offers through this version. Implement domestic trade rules internally; put proposal transport in a separately versioned follow-up rather than silently dropping the rules.
* Refresh from each `decide` snapshot. Optional tree reuse requires a matching canonical state/observation key. A `step` intent alone must not advance an authoritative local root through chance.

No merge or modification of PR #386 is required to start core work. Pin golden fixtures to this head now and recheck the merged protocol before shipping the adapter.

## 3. Experiments performed for this plan

Reproducible code, fixtures and raw observations are in [`experiments/rust-rollout/`](experiments/rust-rollout/README.md). These are deliberately small probes, not a partial production engine.

Hardware/software: AMD Ryzen 9 9950X, 32 logical processors, Windows 11 build 26200; CPython 3.12.14, NetworkX 3.5; rustc 1.90.0 / LLVM 20.1.8, `x86_64-pc-windows-msvc`. Rust uses release optimization, thin LTO, one codegen unit, no third-party crates. Both portable and `target-cpu=native` builds were measured. No CPU affinity, clock locking or system isolation was applied.

### Method and limits

* Generate eight complete four-player random games with seeds 0–7 and `PYTHONHASHSEED=0`; sample 128 positions spread across those games. Export the real 54/72 topology and ownership arrays. All eight games completed below the 1,000-turn limit; together they took 9,315 action ticks.
* For all 128 positions, compare the proposed local road-connectivity rule to Python's `Board.buildable_edges`: **zero mismatches**. In Rust, compare array, recomputed-bitset and cached-node-mask queries. These are geometric road queries, not complete phase/resource-aware legal move generation.
* Compare two independent longest-road traversals on those positions, plus expected-result assertions for a loop with a tail, a blocked loop junction, and a road ending at an enemy building. These tests do not establish complete Catan rule parity.
* Rust: 1,000 warmup calls, nine timed batches per row, 10,000–500,000 operations per batch, input/output `black_box`, result checksums and full raw batch timings. Both halves of 72-edge masks contribute to the observed output. A second native invocation is retained to show run-to-run variation.
* Python: seven batches for small operations, five for full games; initialization/import costs excluded except map/game construction inside the full-game benchmark. The profiler is a separate run. Generation microbenchmarks have warm Python board caches.
* State forks are **synthetic byte-array copies with eight byte mutations**, not implementations of all Catan transitions or a general undo log. Move-buffer probes materialize only road actions. Dispatch uses a synthetic four-variant mixture, including large offer payloads; those offers test layout and are not necessarily legal offers. Packed encoding covers only this probe's variants.
* The fixture corpus is small and repeatedly accessed. These numbers do not model a large transposition table, cold memory, large search trees, weighted rollout policies, feature extraction, Rust JSON parsing or parallel scaling. Nanosecond dispatch measurements are especially compiler-sensitive. Raw ranges are batch ranges, not confidence intervals.

### Python baseline

| Operation | Median |
|---|---:|
| Copy initial state | 11.44 µs |
| Copy a state with 512 history entries | 30.58 µs |
| Copy the sampled state with the longest history | 34.58 µs |
| Generate initial settlement actions, warm | 6.74 µs |
| Generate actions for the selected midgame state, warm | 1.41 µs |
| Copy + execute first legal action, midgame, outer validation disabled | 51.12 µs |
| Render-oriented `GameEncoder` JSON, midgame | 866.22 µs |
| Eight complete random games | 130.12 ms |

The full-game result is about **61.5 games/s and 71,600 action ticks/s** on this machine. The phase-specific generation figures should not be interpreted as an average move-generation cost. The profile places generation at about 0.188 s cumulative out of 0.399 s profiled total and longest-road traversal at about 0.065 s cumulative. String player-key creation and enum hashing are prominent self-time costs. Profiling adds substantial overhead; use unprofiled runs for throughput.

### Rust architecture probes

Primary figures below are portable-build medians; native figures illustrate sensitivity.

| Decision | Measured alternatives | Interpretation |
|---|---|---|
| Board connectivity | Array scan 91.73 ns; recomputed bitsets 32.19 ns; cached expandable-node mask 5.44 ns | Use dense arrays plus bitsets. The cached query excludes maintenance; measure update+query before adding every possible cache. |
| Longest road | Clone-vector DFS 3,672 ns; edge-bitset DFS 321 ns | About 11.4× in this corpus. Start with exact bounded DFS and no copied paths. Native: 3,824 vs 313 ns. |
| Materialize road moves | New `Vec` 55.39 ns; reused `Vec` 8.53 ns; initialized stack array 10.96 ns | Reuse scratch capacity. An eagerly initialized fixed array was not fastest. No claim about `ArrayVec`/`MaybeUninit`: they were not tested. |
| Choose one road | Reused-list generation+selection 9.35 ns; bit-rank selection 7.03 ns | Native: 9.33 vs 5.00 ns. Add direct sampling once it exactly preserves policy probabilities. No RNG generation is timed here. |
| Action layout | Full enum 11 B; packed probe 8 B. Dispatch 1.164 vs 1.201 ns | Use a typed enum as the API. Native reverses the ranking: 1.626 vs 0.886 ns, repeated at 1.624 vs 0.888. Consider packed **search-edge storage** later; there is no universal dispatch winner. |
| Fork 256 B + mutate | Copy 2.86 ns; save/restore eight bytes 2.35 ns | Native approximately ties at 2.39/2.38 ns. Copying a small state is a sound starting point. |
| Fork 512 B + mutate | Copy 5.15 ns; save/restore 3.01 ns | Small absolute advantage for this unusually simple undo operation. |
| Fork 1,024 B + mutate | Copy 7.04 ns; save/restore 3.08 ns | Larger caches have a measurable copy cost. |
| Fork 4,096 B + mutate | Copy 21.23 ns; save/restore 3.15 ns | Keep optional features/history out of state. Reconsider undo if real search needs large state. |

Do **not** divide a Python `Game.copy()` time by a synthetic Rust array-copy time and call that a whole-engine speedup. They do different work. The native repeat also shows timing noise, including a slower vector DFS; architecture choices should survive full-workload benchmarking.

### PR #386 wire-cost probe

This uses the PR's real `decide_message` and serializer loaded against the base engine, plus a persistent Python subprocess that parses each prebuilt JSON message and replies with its first offered action. Startup is warmed out. It does not run the full `StdioPlayer` reader-thread/queue/validation path and does not estimate a Rust parser's cost.

| History entries | Message bytes, including newline | Snapshot construction + JSON | Prebuilt-message echo round trip |
|---:|---:|---:|---:|
| 0 | 6,603 | 42.70 µs | 73.82 µs |
| 512 | 28,732 | 629.87 µs | 291.97 µs |
| 1,229 | 63,231 | 1,453.83 µs | 630.14 µs |

Construction and echo are timed separately. Together they indicate about 0.92 ms overhead at the sampled midgame before useful search. This supports the architectural boundary: **one stdio request per actual bot decision, zero stdio requests per simulated action**. A future protocol can add bounded history/deltas, result-bearing events and request/deadline IDs, but the rollout engine need not wait for those changes.

## 4. Core architecture

### Crates and dependency direction

```text
Python referee / tournament CLI -- JSON lines --> catanatron-bot
                                                    |
                                               catanatron-search
                                                    |
                                               catanatron-core
                                                    ^
RL trainer -- batched binding --> catanatron-python --+

catanatron-bench --> core + policies + optional adapters
```

Create a `rust/` Cargo workspace. `core` owns rules, topology, state, actions and chance; it has no Python, JSON, bot subprocess, UI, global RNG, clock or thread-pool dependencies. `search` owns policy/search state. `bot` owns JSON and lifecycle. The Python binding and benchmark executable depend on core, never the reverse. Initially keep these few crates; use modules rather than many tiny packages.

### Immutable tables and compact mutable state

* `Topology`: dense `NodeId`, `EdgeId`, `TileId` and fixed incident-edge/node/tile tables. Store a 54-bit vertex set in `u64` and 72-edge sets in `u128` initially. Edge endpoints, node neighbors and land-tile vertices are compile-time/exported immutable data. Benchmark `[u64; 2]` against `u128` if generated code or target architecture warrants it; that comparison has not been measured here.
* `BoardLayout`: per-game resource/number/port assignment, dice-to-tile lists, and coordinate/ID mappings. Shared by borrowed reference during simulations, outside copied state. Avoid an `Arc` increment for every child; an owning game/search context can hold it once.
* `Position`: fixed ownership arrays; resource hands `[u8; 5]` per player; development-card counts and start-of-turn eligibility flags; bank counts; piece inventories; knight counts; points/award holders; robber; phase, actor, turn owner, turn count, trade offer and responses. Counts fit bytes for the supported BASE inventories; validate wider input/configuration values and use wider types where bounds are not fixed (for example the turn counter).
* Candidate caches: per-player road masks and road-incident vertex masks, distance-rule-blocked vertices, port rates, longest-road lengths. Ownership arrays remain canonical, caches are recomputable. Measure cache maintenance plus generation plus copying, not just the cached query.
* Target roughly 256–512 B for the initial position and at most about 1 KiB if justified derived data is added. This is a design budget, **not a measured final struct size**. Track `size_of::<Position>()` and allocations in benchmarks.
* Exclude player objects, legal-action vectors, geometry, strings, history, JSON, neural tensors and tree nodes. Pass scratch buffers from the worker/caller. Put optional event history in an outer `GameSession`.

Start with BASE and TOURNAMENT (same geometry), 2–4 players and documented configurations. Add MINI via an active-node/edge/tile mask and topology-specific tables, dispatching outside the action loop. Unsupported arbitrary maps/configurations should fail explicitly until a separately measured generic path exists; do not silently truncate to 54 vertices or slow the common path with unrestricted containers.

### Typed atomic actions

One action is one player intent; resources and chance results are separate concepts. An API sketch:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Roll,
    EndTurn,
    BuildRoad(EdgeId),
    BuildSettlement(NodeId),
    BuildCity(NodeId),
    BuyDevelopmentCard,
    PlayKnight,
    MoveRobber { tile: TileId, victim: Option<PlayerId> },
    Discard(Resource),
    YearOfPlenty { first: Resource, second: Option<Resource> },
    Monopoly(Resource),
    RoadBuilding,
    MaritimeTrade { give: Resource, receive: Resource, rate: u8 },
    OfferTrade { give: [u8; 5], receive: [u8; 5] },
    AcceptTrade,
    RejectTrade,
    ConfirmTrade(PlayerId),
    CancelTrade,
}
```

The actor is determined by the position's decision context; an external command/record carries the actor for validation/replay. Accept/confirm actions refer to the one pending offer in state, avoiding copied offer payloads. The adapter reconstructs Python's larger trade tuples at the boundary. Validate IDs and resource enums during decode. Use ordinary enum layout, no `repr(packed)`, boxes, trait objects or unsafe transmute. Rust's layout is not the wire format or the RL action index.

Keep discards one card at a time, as this checkout does; this caps immediate branching at five instead of enumerating all discard multisets. Keep Road Building as card play followed by zero/one/two legal free-road decisions, and Knight as card play followed by robber/victim selection. Do not turn all legal same-turn sequences into gigantic compound actions. Canonicalize order-insensitive Year of Plenty pairs; preserve the one-card case and the selected compatibility profile's availability rules.

### State machine, generation and application

Replace overlapping phase booleans with a typed phase carrying its required data: setup settlement/road with forward/reverse seat order and last settlement; pre-roll; post-roll; discard queue; robber selection with its resume phase; free-road sequence with remaining count/resume phase; trade responses; choosing an accepter; terminal. Actor and turn owner are different during discards/trades. Preserve whether a knight was played before rolling. Consume the development-card allowance at card play and enforce newly bought card timing.

Conceptual API (signatures are a design sketch):

```rust
generate_actions(layout, position, output_buffer); // read-only position
validate_action(layout, position, actor, action) -> Result<(), IllegalAction>;
apply_checked(layout, position, actor, action) -> Result<Transition, IllegalAction>;
apply_generated(layout, position, action) -> Transition; // crate-private fast path
chance_outcomes(layout, position, pending, output_buffer);
resolve_chance(layout, position, outcome) -> Transition;
sample_action(layout, position, policy, rng, scratch) -> Action;
```

`apply_generated` means “already known legal in this position,” not memory-unsafe. Use debug assertions and keep the unchecked entry point internal. Checked mutation validates fully before changing state. `Transition` reports the next decision, pending chance, terminal result and optionally an event; it does not generate a list, serialize state or invoke an observer. Search calls it directly. A thin convenience `GameSession.step()` can combine validation, chance sampling and optional logging for ordinary callers.

Use one move-emission implementation with a statically dispatched sink/visitor so it can fill a reused `Vec<Action>`, count categories, or produce an RL mask. Start with the reused vector because it won the probe and has safe growth. A later fixed-capacity buffer needs a proven bound for **all supported phases/configurations**, including trade responses and future variants, plus an explicit overflow path. Never silently truncate moves.

`generate_actions` is the enumerable menu. Parameterized domestic offers are separately validated legal proposals, matching Python's existing split. Policy-level proposal generation must be explicit; exhaustive enumeration of all offer/request combinations is not a hot-loop requirement. Also separate heuristic pruning from rules: a policy may choose best-rate trades, but an engine should not call omitted actions illegal when its selected rules profile permits them.

### Board kernels

For road legality, an empty edge is reachable through either an own building or an unoccupied vertex incident to an own road. An enemy building blocks extension through that vertex, but does not prohibit approaching it from the other endpoint. This local predicate matched the sampled Python states and avoids connected-component containers entirely.

For settlements, combine an empty-and-distance-legal node mask with own road connectivity; setup bypasses connectivity but still enforces distance. For cities, select own settlements and check cost/piece availability. Update affected masks on placement; city upgrades do not change ownership/connectivity. An opponent settlement invalidates affected players' connectivity and longest-road data, not just the current player's cache.

Longest road is an **edge-simple trail**, not a vertex-simple path, a graph diameter or union-find component size. A player's 15-road inventory bounds the search. Start DFS from relevant vertices, mark used edges in a bitset, allow revisiting vertices, and terminate continuation at enemy buildings while counting the incoming road. Keep an independent slow oracle. Recompute only for a road addition or a settlement that can interrupt a road; implement tie/incumbent and minimum-length award rules separately. Avoid clever extension-only shortcuts until loops, branches, blocked junctions and award removal are covered.

For dice production, first use preindexed matching tiles and tiny `[player][resource]` demand arrays; suppress the robber tile and resolve bank shortages per resource before transferring cards. A cached `[dice_sum][player][resource]` table may help, but adds around 200 bytes plus maintenance/copy cost. **No production-cache experiment was performed here**: benchmark it against tile-local calculation before adopting it. Resource transfers, monopoly, maritime trades and payments should be bounded array operations.

### Chance, reproducibility and hidden information

Distinguish intent from `ChanceOutcome`: dice pair/sum, stolen resource or no theft, development-card type. Use deterministic `resolve_chance` for fixtures and sampled resolution for rollouts. Keep chance nodes out of the player-action catalogue.

* Dice: 36 equiprobable ordered pairs, or 11 sums weighted `1,2,3,4,5,6,5,4,3,2,1` over 36. Group by sum when it has identical state effects; retain the pair only when exact replay/logging needs it. Seven triggers the discard/robber sequence, never production.
* Theft: sample resource by its count in the victim's hand, not uniformly over distinct types. Enumerate only possible outcomes with integer weights.
* Development draws: sample without replacement from remaining counts for client-view hypotheses. An authoritative ordered-deck import must retain the supplied order in its game context and resolve accordingly; do not claim count-only state preserves that exact future sequence.
* Do not multiply chance branches eagerly during random rollouts. Expose enumeration to expectimax/MCTS separately and group genuinely equivalent outcomes only when observations/rewards also agree.

Pass RNG state explicitly through a sampler/rollout context, separate from policy randomness. Choose and version a small generator after speed/distribution/reproducibility checks; sample bounded integers without modulo bias. Derive independent deterministic streams from `(run_seed, game_id/index, rollout_id, stream_kind)` so scheduling and rejected policies do not perturb other games. Copying a position neither clones a global random stream nor implicitly advances a parent stream. Exact session checkpointing stores RNG state separately; deterministic replay supplies outcomes.

Search over a fully specified `Position` and import from an `Observation` must be separate operations. For future fair play, sample hidden hands/dev cards consistent with public counts and the observer's information, then run many local simulations. Never call `client_view` an information-safe observation or use a sampled hidden world as a universally shared information-set tree key.

### Copying, search and rollouts

Implement copy-and-apply first. A rollout copies its root once and mutates that copy until termination/cutoff. It does **not** copy before every forward step. A depth-first search can keep positions on a depth-indexed stack; a broad tree can retain edges/statistics and reconstruct states from periodic checkpoints. Avoid eagerly storing every child state/chance branch as Python's expansion does.

Use arena indices for search nodes/edges, compact counters/statistics, and independent per-worker scratch/RNG. The typed action can later encode into a compact search-edge representation if tree-memory and end-to-end benchmarks justify conversion. This leaves the rule API readable while optimizing the large structure that actually benefits from packing.

Introduce undo only if realistic search shows copying is material. Compare copy-and-apply against make/unmake across roads, settlements, payouts, robber, monopoly and trade—not just eight-byte mutation. An undo record must restore award incumbents, phase/actor, dev eligibility and every cache or invalidate it correctly. Require `unmake(make(s,a,o)) == s` and equal recomputed hashes/legality. The current probe does not justify committing to this complexity up front.

Start random policies by choosing uniformly from the generated menu. A later direct sampler may count legal choices per category, select a category **proportional to its count**, then choose an action within it; uniform choice of action types changes the policy. Weighted policies need weighted totals. Verify exact enumeration probabilities on small states before comparing rollout strength/speed. Measure RNG and policy cost in complete rollouts.

Use canonical incremental hashes only after correctness: include map/rules identity, players, hidden counts when appropriate, phase, pending trade/chance, award incumbents, dev timing, and any turn-limit state. Exclude irrelevant history. Position and information-set keys serve different purposes. Verify against a slow canonical hash and retain collision verification data in transposition tables.

### Parallelism and RL

Make single-core performance the first target. Then parallelize independent rollouts/environments with one position/scratch/RNG stream per worker; aggregate statistics in batches. Avoid a mutex or atomic update per simulated action. Benchmark scaling at 1/2/4/8/16 physical cores and then SMT, memory use and worker startup separately. Test deterministic results across scheduling changes with fixed rollout IDs.

Provide a later PyO3 binding with `reset_many`, `step_many`, `observe_many` and `rollout_many`, contiguous output buffers and one language crossing per batch. Release the GIL while Rust runs. Avoid building Python `Game` objects, JSON messages or Python dictionaries for every simulated action. Keep policy inference outside the pure rules crate; batch neural evaluations across environments/search leaves. Start with independent positions (AoS); adopt SoA/SIMD/GPU designs only if batch measurements identify a benefit. Branchy variable-length Catan games do not establish a GPU advantage by themselves.

Version the RL action space and observation features. Supply an adapter for existing Gym catalogue indices; do not use Rust discriminants or PR list order as policy indices. Carry variable legal menus/masks, acting-player identity, reward perspective, and distinct terminated/truncated flags. Treat a per-card discard as one step consistently, and document any alternative decision aggregation. Test reset after terminal/cutoff and consistent reward accounting with Python.

## 5. Correctness before speed claims

Create a pinned **Catanatron compatibility profile** first. The Python oracle has known moving behavior; do not confuse observed behavior with an unexamined official-rule specification. In particular, review bank-shortage semantics, Year of Plenty single-card availability, best-rate maritime generation, victory timing and longest-road blocking/award ties. Main's `winning_color()` scans every player, and `generate_playable_actions()` does not itself stop at a winner. Define terminal behavior explicitly and record any intentional divergence in fixtures. Never silently preserve or silently fix an oracle bug.

Validation layers:

1. Export authoritative JSON snapshots, canonical maps, sorted action menus and action/outcome trajectories from the pinned Python revision/PR schema. Include each phase, every action variant, 2/3/4-player setups, BASE/TOURNAMENT and later MINI. Treat wire actions as values; catalogue order is a separate compatibility check.
2. For each deterministic transition, compare canonical semantic state and the next action set. For stochastic transitions, feed the same forced outcome into both engines and compare all legal outcomes/probability weights. Use the separate records to replay; do not compare RNG sequences across languages.
3. Port targeted tests: setup snake order and second-settlement resources; distance and piece limits; enemy endpoint blocking; loops/branches/junctions and longest-road ties/removal; largest army; new dev cards and one card per turn; pre-roll knights; zero/one/two free roads; seven and multiple discarding players; theft with empty hands; depleted banks; ports; all trade phases; terminal and cutoff behavior.
4. Assert resource conservation, fixed inventories, unique ownership, no invalid IDs, legal actor/phase, old/new dev consistency, awards/points, and cache equality with recomputation. Property-test generated sequences and checked/fast-path equality. Reject malformed snapshots without partially mutating the engine.
5. Run many seeded differential trajectories with controlled outcomes, retaining minimized failures. A rules discrepancy is triaged and fixed/profiled explicitly before performance comparisons. Add rare/adversarial states beyond random play; the current 128-state probe is only initial evidence.
6. Protocol conformance: hello/version, multiple games in one process, map caching, skipped forced decisions, chance resynchronization, exact action echo, newline/flush, EOF, malformed input, configured deadline and no extra stdout. Then complete mixed Python/Rust matches through the real `StdioPlayer` after #386 merges.
7. For search/RL, verify parent immutability, clone independence, deterministic worker streams, policy sampling distributions, legal masks, terminal/truncation rewards and no hidden-information leakage in fair mode.

## 6. Performance contract and next experiments

The primary scoreboard is **correct complete in-process games/s and action transitions/s**, plus rollouts/s from fixed midgame states. Always report both games and actions, winner/truncation counts, mean/quantiles of game length, rules/map/player count, policy, seed set, CPU, compiler flags and threads. Games/s alone can improve by changing policy, shortening games or abandoning hard trajectories.

Initial acceptance target: at least **10× the same-machine Python baseline** on both complete random games and fixed-state rollout work, with matching rules/policy and all correctness gates passing. A useful first optimization objective is at least **1 million action transitions/s on one core**; 10 million/s is a stretch investigation target, not a promise. Rebenchmark Python after #386 and relevant fixes merge. Do not derive complete-game throughput from the tiny board probes.

Maintain separate measurements for:

* generation by phase, checked vs generated application, explicit chance resolution, board cache maintenance, longest-road worst cases, clone/fork and feature extraction;
* fixed-corpus rollout throughput and latency, full games including initialization, random and weighted policies, 2- and 4-player games, normal and cutoff-heavy trajectories;
* copy-and-apply search with real branching/depth, transposition-table hit rate, bytes per node, allocations per transition, and cold/large working sets;
* protocol snapshot construction, Rust import/parse, end-to-end stdio decision latency and search-only time; binding crossings and batch sizes separately;
* one-core speed first, then physical-core and SMT scaling. Native/LTO/PGO builds get separate rows from portable releases.

Use a fixed machine for regression budgets; start by flagging sustained >5% regressions across repeated runs, not noisy shared-runner failures. Require zero steady-state heap allocations for the supported in-process random rollout path after scratch initialization; measure with an allocation counter, not inference from types. Profiles/hardware counters can guide later inlining, cache, branch and SIMD work. Avoid unchecked indexing/unsafe code until a measured hot spot and a documented invariant justify it.

Next architecture experiments, in order:

| Experiment | Adoption rule |
|---|---|
| Cached connectivity vs local recomputation including placement and copying | Retain caches only if complete rollout/search time improves; test enemy settlement invalidation. |
| Dice-local payout vs incremental production table | Include setup/build/robber maintenance, copy cost, depleted banks and feature reuse. |
| Exact simple longest-road DFS vs local 15-edge masks/pruning | Equal results against oracle on adversarial max-road graphs; improve p95/p99 as well as average. |
| Reused vector vs safe fixed-capacity/visitor generation vs direct sampling | All legal phases, bounded capacity and identical policy distributions; measure full generation+choice+application. |
| Copy vs undo at actual `Position` size in MCTS/depth search | Adopt undo only if end-to-end benefit pays for restoration complexity. |
| Enum vs packed search-edge storage | Large tree working set and real action distribution; include encode/decode and node memory. |
| AoS vs batched SoA, feature/inference batching | End-to-end training samples/s, including resets and variable episode length. |
| Portable vs native vs PGO, then worker scaling | Preserve correctness and reproducibility; train PGO on a separate corpus. |

To substantiate “fastest,” establish a dated comparison suite against available maintained Catan engines, pin their revisions/configurations, run them on the same hardware under equivalent rules/policies, and publish raw results and mismatches. No external engine comparison was run for this plan; no worldwide ranking is claimed. Existing open Rust rewrite PRs #288 and #304 are useful candidates for later comparison, not assumed performance baselines or source architecture for this fresh implementation.

## 7. Implementation sequence and exit criteria

| Stage | Deliverable | Exit criterion |
|---|---|---|
| 0. Freeze contracts | Rules/profile document, pinned protocol/schema samples, map tables and Python fixture exporter | Every phase/action has a sample; disputed rules identified; benchmark commands reproducible. |
| 1. Small correct kernel | `core`: typed IDs/actions/phases, fixed state, setup/builds, costs, ports and exact longest road | Deterministic board cases and cache invariants pass; generation/application are independently callable. |
| 2. Complete simulation | Dice/bank/robber/discards, all development cards, domestic trade, awards, terminal/cutoff and explicit RNG/outcomes | Complete BASE games; controlled-outcome differential corpus passes; random-policy probabilities checked. |
| 3. Measure the engine | In-process random/weighted policy runners, fixed-state rollouts, profiling/allocation counters | Publish same-machine Python comparison; zero steady-state rollout allocations; meet initial 10× target or profile and resolve bottlenecks before claiming success. |
| 4. Integrate stdio | `bot`: v1 handshake/import/action mapping/deadline; simple random policy then Rust rollout search | Mixed-process games finish with zero fallback/illegal-action incidents; imported roots and offered actions match; merged-PR conformance passes. |
| 5. Search optimization | Arena search nodes, chance sampling/enumeration, independent rollouts and optional TT | Equal-budget playing-strength tests plus nodes/rollouts per second; benchmark copy vs undo and packed edges before adoption. |
| 6. RL and scale | Batched Python API, action/feature codecs, reproducible parallel environments; MINI support when needed | Batched/scalar parity, documented semantics, no fairness leaks, samples/s including feature/inference costs and multi-core scaling reported. |
| 7. Competitive performance | Large/adversarial corpus, PGO/native variants and comparable external-engine harness | Reproducible public evidence for any “fastest” claim, scoped to measured workloads. |

The first useful milestone is a correct Rust core that finishes complete games in-process. The stdio milestone then makes its bot directly comparable to existing Python bots without coupling rollout speed to JSON or Python. The long-term RL/search interface grows out of that same core, so there is only one Rust implementation of the rules to validate and optimize.
