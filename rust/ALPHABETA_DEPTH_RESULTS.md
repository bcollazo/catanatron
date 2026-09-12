# Alpha-beta depth tournament

Date: 2026-09-08

The Rust alpha-beta agents were compared in full four-player BASE games with a
20 ms search budget per decision. Games were grouped into four-game blocks.
Within each block the board and RNG seed were held fixed while the four search
depths rotated through all four seats.

## Results

| Field | Games | Depth | Wins | Win rate |
| --- | ---: | ---: | ---: | ---: |
| 1, 2, 3, 4 | 2,400 | 1 | 36 | 1.50% |
| 1, 2, 3, 4 | 2,400 | 2 | 540 | 22.50% |
| 1, 2, 3, 4 | 2,400 | 3 | 891 | 37.13% |
| 1, 2, 3, 4 | 2,400 | 4 | 933 | 38.88% |
| 3, 4, 5, 6 | 806 | 3 | 182 | 22.58% |
| 3, 4, 5, 6 | 806 | 4 | 214 | 26.55% |
| 3, 4, 5, 6 | 806 | 5 | 213 | 26.43% |
| 3, 4, 5, 6 | 806 | 6 | 197 | 24.44% |

The first run completed in 1,596.28 seconds. The second stopped at its
1,800-second safety limit after finishing 806 games. Neither run truncated a
game. Total simulation time was 3,397.79 seconds (56 minutes, 38 seconds).

In the first field, depths 1, 2, and 3 are clearly separated. Depth 4's 1.75
percentage-point lead over depth 3 is not statistically significant. In the
deeper field, a chi-square goodness-of-fit test against equal win shares gives
approximately chi-square=3.42 with 3 degrees of freedom (p=0.33). No pair among
depths 3 through 6 is significantly different at the 5% level.

The evidence does not show a statistically reliable inversion where a deeper
agent is weaker. It does show diminishing returns: under a shared 20 ms ceiling,
increasing the configured depth beyond 3 or 4 produced no detectable strength
gain. These conclusions apply to the current evaluator, action ordering, and
deadline behavior.

## Reproduction

```powershell
cargo run --release --manifest-path rust\Cargo.toml -p catanatron-bench --bin alpha-depth-tournament -- 2400 20 3000 1,2,3,4
cargo run --release --manifest-path rust\Cargo.toml -p catanatron-bench --bin alpha-depth-tournament -- 1200 20 1800 3,4,5,6
```

Arguments are `GAMES BUDGET_MS WALL_SECONDS DEPTHS`. `DEPTHS` must contain four
distinct positive comma-separated depths.
