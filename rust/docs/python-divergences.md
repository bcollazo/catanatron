# Python rule divergences found by the Rust rewrite

The executable reproductions live in
`tests/test_known_rule_divergences.py`. They assert the intended rule and are
marked `xfail(strict=True)` while Python still has the old behavior. Run them
normally to keep the suite green, or add `--runxfail` to see each bug fail at
its exact assertion.

## D001 — proposer revisited during trade responses

The proposer must be skipped while the other seats answer:

```text
seat 0                 seat 1                    seat 2
first responder  --->  PROPOSER (skip)  ------>  next responder

Python:  0 ----------> 1   (incorrectly asks the proposer)
Correct: 0 ------------------------------------> 2
```

## D002 — road ending at an opponent building

An opponent building blocks passage through its vertex, but the incoming road
still counts:

```text
ORANGE road
35 -------- 36 -------- 37 -------- 38 [BLUE settlement]
     edge 1       edge 2       edge 3

Python length: 2
Correct length: 3   (the trail ends at 38)
```

## D003 — stale/undercounted branching-road cache

The side spur does not invalidate the five-edge trail:

```text
          21
           \
0 -------- 20 -------- 19
             \
              22 -------- 23 -------- 52

valid trail: 21 - 19 - 20 - 22 - 23 - 52  = 5 edges
cached Python value observed by differential test: 4
```

The minimal test independently calculates the five-edge path, then exposes the
stale value captured at the production cache/award boundary. The committed
differential fixture preserves the originating live observation.

## D004 — incumbent displaced by a tie

```text
before split: ORANGE=7 (holder), RED=5
after split:  ORANGE=5 (holder), RED=5

Python:  award moves to RED
Correct: ORANGE retains the award because the incumbent is tied at >= 5
```

## D005 — award below the five-road threshold

```text
before split: RED=4 (stale holder), BLUE=4
after split:  RED=2, BLUE=4

Python:  awards BLUE at length 4
Correct: no holder; Longest Road requires at least 5
```
