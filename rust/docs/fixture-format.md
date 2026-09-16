# Rust fixture format (v2)

Fixtures are canonical JSON or JSONL records generated from the pinned Python
rules source. A transition record contains `fixture_version`, `case_id`,
`source_revision`, `rules_profile`, `before`, `actor`, intent `action`, an
optional stochastic `outcome`, `after`, `legal_before`, `legal_after`, and
`status_after`. Intent and outcome are separate: a roll and development-card
purchase have `value: null` in the intent; their concrete result is in
`outcome`.

State snapshots contain only semantic fields: concrete map and port assignment,
active seats, owned roads/buildings, bank/hands/development counts, piece
inventories, graph-derived award state, robber position, actor/turn owner, a
typed phase with resume payload, pending trade/responses, and turn count. The
legacy Python prompt and timing flags remain diagnostic fields; importers use
the typed `phase`. Caches, UUIDs, history, and RNG state are intentionally
excluded. Enum values become strings, coordinates/edges become JSON arrays,
and action menus are compared canonically without depending on ordering.

The corpus covers all 18 Python action variants, all reachable decision prompts,
BASE/TOURNAMENT, and 2/3/4 seats. The E08 conformance runner imports both
boundaries, compares both legal menus, applies the intent and forced chance
outcome, and compares the resulting typed state and status.
