# Local upstream performance candidate

Prepared September 11, 2026 UTC. Branch `perf/native-evaluator-speedups` is based
on latest upstream `ecf931181b9a65bb4116a2153fb78c16f1438e00`, checked directly
against GitHub. The PR is prepared for submission with the user's approval.

GitHub has no published releases or tags. PyPI's latest release is
[3.2.1, published July 17, 2022](https://pypi.org/project/catanatron/3.2.1/).
Current GitHub source declares 3.3.0 and is the relevant base for a future PR.
HexSet's previous benchmark pin, `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`,
is two commits behind this base. The two newer commits introduce per-game RNG
and the bot registry/lifecycle API; this branch preserves both, including the
intentional shared RNG reference in State.copy. Historical HexSet results
remain tied to their original pin and must not be relabeled as latest-upstream
measurements.

## Changes and prior work

The first commit is cherry-picked from [#382](https://github.com/bcollazo/catanatron/pull/382)
by Honesty Bot, preserving its original authorship. It removes the duplicate
production-feature calculation. That contribution is included here explicitly;
we do not claim it as new work. This PR can follow #382 if it lands first.

The subsequent change adds structural state/board copies, perspective-only hand
and reachability extraction, and bounded board-feature caches owned by individual
AB/ValueFunction players. Caches clear when maps or seating change. Dynamic hands,
VP, development cards, army and longest road remain live reads. Full public
feature extraction retains its default behavior. No global patch context,
HexSet dependency, heuristic weights, action ordering, pruning or search deadline
is changed.

Related performance work: [#371](https://github.com/bcollazo/catanatron/pull/371)
optimizes MCTS playouts, enum hashing and longest-road traversal. This change
keeps enum representations and the road algorithm unchanged. The maintainer's
Rust work in [#387](https://github.com/bcollazo/catanatron/pull/387) is separate
from this Python implementation.

The tested implementation is byte-identical to the measured source at local
commit `48aeb7d7512aade72df2a13a17a94e2ad667542c`; commit history was reorganized
to preserve #382 authorship before submission. The benchmark retains a frozen
pre-change scalar oracle. The headline timings include the benefit of #382;
they are not an attribution to the additional work alone.

## Latest-upstream paired check

Pinned Python 3.12.14 Wintermute runtime, one CPU, fresh interpreter per arm,
PYTHONHASHSEED=0, one ValueFunction and three AB:2 players, standard rules.
Arm order alternates by seed. These games use Catanatron alone.

| Mode | Four clean games, wall seconds | CPU seconds |
| --- | ---: | ---: |
| unmodified latest upstream | 43.290 | 43.273 |
| local candidate | 12.918 | 12.911 |

On clean seeds 670100000..670100003 the local candidate delivers **3.35x**
full-game throughput (70.16% less wall time), with identical ordered action
hashes, winners and VP in every pair. This is a four-game sample, not a broad
performance guarantee or an isolated attribution to either commit.

Separate audit seeds 670000000..670000001 match **54,372** scalar leaf values,
ordered leaf digests/counts, all actions and outcomes against unmodified latest
upstream. Every evaluated scalar is also checked against the frozen original
formula. No audited deadline cutoff was observed. Audit timings are excluded
from the speed claim; behavior can still differ on other deadline-limited
positions because a faster engine can finish more search.

## Tests and reproduction

182 focused tests pass. The broader local Python 3.14 run passes 328 tests:

```sh
python -m pytest -q --benchmark-disable \
  --ignore=tests/integration_tests/test_play.py \
  --ignore=tests/integration_tests/test_server.py --ignore=tests/web \
  -k 'not test_render_rgb_array'
```

The excluded tests need unavailable pandas/web/pygame dependencies. An initial
broader run also hit one 200ms stdio-bot startup timeout; the complete supported
selection above passed on rerun. Full optional UI/data integration validation
remains for an environment with those dependencies before an upstream PR.
Black 25.11.0 and git diff --check pass for changed files.

`run_comparison.py` records the exact trial plan. It expects clean baseline
and candidate exports at `/study/base` and `/study/fast`, and writable `/out`.
It selects each source through PYTHONPATH and invokes the candidate's
`examples/benchmark_value.py` from a fresh interpreter. `comparison.json`
contains every child result and source fingerprint. `runtime.json` records the
immutable image, read-only source mount, CPU limit, environment and exit.

PR title: **Reduce state-copy and native evaluator overhead (building on #382)**.
