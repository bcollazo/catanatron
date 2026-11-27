# Paired challenge against Catanatron's AlphaBetaPlayer

The candidate won **96 of 128 games (75.0%)** against the stock depth-2 AlphaBetaPlayer on 64 fresh boards. It won 47 of 64 games moving first and 49 of 64 moving second. All games finished. The 95% percentile bootstrap interval for the candidate win rate was 67.19% to 82.81%, resampling whole boards to keep the two games on each board together.

| Candidate's wins on a board | Number of boards |
| --- | ---: |
| Neither game | 4 |
| One game | 24 |
| Both games | 36 |

At the pinned revision, the Catanatron project listed depth-2 AlphaBetaPlayer first and invited reproducible challenges. This attachment records a local comparison under that project's 1v1 benchmark. [Catanatron project benchmark](https://docs.catanatron.com/advanced/making-catanatron-stronger)

## Confirmation on 500 fresh boards

With the policy and weights frozen, the candidate won **721/1,000 games (72.1%)** on 500 additional boards. It won 371/500 moving first and 350/500 moving second. All games finished. Resampling the 500 board pairs gives a 95% interval of 69.3% to 74.9%. The candidate lost both games on 42 boards, split 195 and won both games on 263. The board seeds and initial setup digests are disjoint from those in the original 128-game test.

The [full confirmation record](artifacts/catanatron-duel-1000.json.gz) contains the runner's JSON output compressed with gzip. To reproduce the confirmation after installing the dependencies below, run these commands in this directory.

```sh
.venv/bin/python catanatron_duel.py --boards 500 --seed 27062041 --workers 4 --output artifacts/confirmation.json
gzip -dc artifacts/catanatron-duel-1000.json.gz | cmp - artifacts/confirmation.json
```

The uncompressed confirmation report has SHA-256 `c9d5d18482ee150ad69521b40b566f643cd9c63052e342e7393a76ca6b96f346`.

## Run from the repository

From the repository root, enter `examples/tactical_player`. Use Python 3.12 and the pinned Catanatron revision for the paired replay. The runner checks the installed engine digest before starting.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python catanatron_duel.py --boards 64 --seed 948561723 --workers 4 --output artifacts/replay.json
cmp artifacts/catanatron-duel.json artifacts/replay.json
```

The bundled reference and an independent replay on the original machine matched byte for byte. The report records interpreter and NetworkX versions, so a byte comparison includes that metadata. The runner reads the adjacent `agent.py` and `weights-3p-tactical.json`. The JSON report includes every board seed, initial setup digest, seat, player color, winner, both final scores, turn count and domestic offer count.

## Play conditions

- The referee was the project's unmodified `Game.play()` with two seats, its base map and spiral number placement, a target of 10 VP, a discard limit of 7, the ordinary robber and the native 1,000-turn limit. Both agents used the same referee. The opponent was instantiated as `AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=2))` with all other settings at their defaults.
- For each initial seed, the two policies exchanged seats. The runner verifies that the board, development deck and starting random generator state match between the two games. Native search shares the referee's random generator. Different decisions may therefore lead to different later dice in the two legs.
- The candidate carries over weights trained for three-player Catan. It suppresses domestic offers for this two-seat comparison because the stock tree search at the pinned revision cannot evaluate domestic trade replies. Bank and port trades are available. Neither policy made a domestic offer in these games. Three-player trading performance requires a separate test. [Upstream domestic trade search discussion](https://github.com/bcollazo/catanatron/pull/399)
- Seed 948561723 determines all 64 starting boards. Four fresh worker processes used Python hash seed 0. The confidence interval used 10,000 resamples of the 64 complete board pairs, with bootstrap seed 175983107. The repeated run contributed no additional observations.

## File identity

| File | SHA-256 |
| --- | --- |
| `agent.py` | `938088b1cff75b2add60e54f06cf12651ef078c0845914e8a88af3bb2f71def2` |
| `weights-3p-tactical.json` | `71bab44d172e818e3c40dfb9308186c97bfd4297794ae1667d6fbd0d8b2b1b82` |
| `catanatron_duel.py` | `b1e8a5924d47fd03bace8673277dfc7e3fe30a8a78651408d2c657a8e50379b3` |
| `artifacts/catanatron-duel.json` | `27ed3758b51c9979ad0ee8121d01608c23f7a367e4f497cd8a5b62293c9584fe` |

## Use with catanatron-play

From the same directory, the following command loads the frozen two-player policy with stock AlphaBeta as its opponent.

```sh
.venv/bin/catanatron-play --bot TACTICAL=./duel_player.py --players TACTICAL,AB:2 --num 4
```
