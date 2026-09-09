# Perfect-information observation schema v1

`catanatron_rust.Batch` returns an owned, C-contiguous `numpy.int16` matrix named `features`.
Each row has 194 columns in this fixed order:

1. player count, acting-player index, robber tile index (3 columns);
2. resource bank in WOOD, BRICK, SHEEP, WHEAT, ORE order (5);
3. all 54 building slots (empty `0`, settlement seat `1..4`, city seat `5..8`);
4. all 72 road slots (empty `0`, owner seat `1..4`);
5. four fixed player slots, each containing five hand counts, five development-card counts,
   three remaining-piece counts, played-knight count, and played-development-card flag (60).

Inactive MINI geometry and inactive player slots remain zero. This schema deliberately exposes
perfect information, including every hand and development-card holding. It is not claimed to match
Python `create_sample`; fair hidden-state observations and belief sampling are outside v1.

The action schema is independently versioned. Dynamic menus use ragged `uint64` IDs and offsets;
Gym compatibility uses the exported stable catalogue and a dense `uint8` legality mask.
