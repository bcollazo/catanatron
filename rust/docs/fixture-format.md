# Rust fixture format (v1)

Fixtures are canonical JSON or JSONL records generated from the pinned Python
rules source. A transition record contains `fixture_version`, `case_id`,
`source_revision`, `rules_profile`, `before`, `actor`, intent `action`, an
optional stochastic `outcome`, `after`, `legal_before`, `legal_after`, and
`status_after`. Intent and outcome are separate: a roll and development-card
purchase have `value: null` in the intent; their concrete result is in
`outcome`.

State snapshots contain only semantic fields: concrete map assignment, active
seats, ownership, bank/hands/development counts, timing flags and piece
inventories, awards, robber position, actor/turn owner, prompt/trade state,
and turn count. Caches, UUIDs, history, and RNG state are intentionally
excluded. Enum values become strings, coordinates/edges become JSON arrays,
and action menus are canonicalized by sorting their JSON representations.

The E02 exporter currently emits deterministic initialized states and bounded
sample traces. Its manifest reports actual coverage; it must reach the guide's
full action/phase matrix before E02 can be checked off.
