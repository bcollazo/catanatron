# `rust-v1` rules profile

`rust-v1` is a compatibility profile derived from Python rules commit
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`. It supports BASE and TOURNAMENT
initially (TOURNAMENT shares BASE geometry); MINI is deferred to E12.

Resource order is `WOOD, BRICK, SHEEP, WHEAT, ORE`; development-card order is
`KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT`. The bank
holds 19 cards of each resource. Each player starts with 15 roads, five
settlements, and four cities. Development-card counts are 14/2/2/2/5.

Costs are road `[1,1,0,0,0]`, settlement `[1,1,1,1,0]`, city
`[0,0,0,2,3]`, and development card `[0,0,1,1,1]`.

The initial release follows the execution guide's explicit compatibility
policies: per-resource bank-shortage suppression; only the best available
maritime rate per given resource; Python's Year of Plenty candidate algorithm;
start-of-turn non-VP eligibility masks; winner scan over active seats after a
completed intent/chance transition; Python turn counting; and optional
friendly-robber filtering with fallback. Longest-road and domestic-trade
corrections are documented as named divergence fixtures before they are
implemented. `D001-domestic-trade-proposer-revisited` corrects the pinned
Python response-advance behavior: each other seat responds exactly once, and
the proposer never responds to its own offer.

This profile is not a claim of universal official-rule conformance. The
canonical fixture format and each accepted divergence will be added before the
engine consumes production fixtures.

See [`python-divergences.md`](python-divergences.md) for the five named Python
reproductions and compact ASCII diagrams.
