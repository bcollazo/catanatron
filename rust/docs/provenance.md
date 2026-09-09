# Rust rollout-engine provenance

Recorded 2026-09-07 for the local implementation session.

## Checkout and compatibility sources

* Implementation checkout: `plan/rust-rollout-engine` at
  `5cc406a4032b3a7f6971b4fc0735663d2e0ca7e5`
  (`docs: plan Rust rollout engine with executable handoff guide`).
* Rules baseline: `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`
  (`fix: block road extension through opponent's settlement endpoint (#377)`).
  This is also the current local `main`/`origin/main` tip.
* Protocol/schema baseline: PR #386 head
  `5149b1869ba6318a2f2e3ef3925915576a433286`
  (`Rewrite the custom-bot docs for the new flow`). The commit is available
  locally, but is not reachable from local `main`; no newer merged #386 was
  found in the local history. Core/fixture work therefore uses the rules
  baseline; stdio certification remains gated on this pinned protocol source
  or its recorded merged successor.

## Tooling and host

* OS reported by `Get-ComputerInfo`: Windows 10 Home, version 2009.
  WMI CPU queries were denied in this execution environment, so CPU model and
  core count are intentionally not claimed here.
* `rustc 1.90.0 (1159e78c4 2025-09-14)`, LLVM 20.1.8,
  `x86_64-pc-windows-msvc`.
* `cargo 1.90.0 (840b83a10 2025-07-30)`.
* Python: `C:\\dev\\catanatron\\.venv\\rust-engine\\Scripts\\python.exe`,
  Python 3.12.14, with the editable local `catanatron` package and
  `networkx==3.5`. The system `python` is an unconfigured pyenv shim and was
  not used.

## Commands actually run

* `git status --short --branch`
* `git rev-parse HEAD`
* `rustc -vV`
* `cargo --version`
* `cargo run --release --offline --manifest-path experiments/rust-rollout/Cargo.toml`
  — passed its 128-fixture assertions.
* `python experiments/rust-rollout/baseline.py` with `PYTHONHASHSEED=0`
  — passed with `corpus_size=128` and `road_query_mismatches=0`.
* Editable core import probe — passed (`catanatron`, `networkx==3.5`).

The production Rust workspace has not been created at this point. The
experiment remains separate under `experiments/rust-rollout/`.
