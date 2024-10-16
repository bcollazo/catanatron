# catanatron_rust

This is a rewrite of Catanatron Core in Rust.

## Usage

```
cargo run
cargo test
cargo bench
```

### Benchmarking

To debug speed

```
CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --release --root
cargo flamegraph --dev --root
```

## Performance

We will make sure MoveGeneration is the one that only creates
valid moves. That way, applying moves don't need to validate.

### State as Vector

Small experiments (see copy_benchmark) showed its
faster to copy a small vector than a larger array. And cloning
a higher abstracted / complex object like a struct would take
a lot more! So going with a compact Vector to represent it.

## Actions as BitFields

Reading GYM actions, that are 289, we prob need at least 9 bits.
To represent Actions as a bitfield number, we would need:

- 2 bits for the color (4 colors). Or 3 bits for up to 8 colors.
- 5 bits for type of action (18 types)
  its reasonable to say that the numbers in the trade freqdecks are 0-4, so we can use 3 bits for that.
  biggest value is that of CONFIRM_TRADE, which is 11 3-bit numbers, so 33 bits.
- 33 bits for the value of the action.
- if we were to remove trading, we would have 5 bits for the value of the action.
  because of we would need to represent just MAX of:
  - 6 bits: 2 numbers of up to 6 (for the dice roll), which is 3 bits each. So 6 bits.
  - 11 bits: tiles id (up to 19), which is 5 bits (+ 3 bits of color + 3 bits of resource) (MOVE_ROBBER)
  - 7 bits: BUILD_X (up to 72 edges), which is 7 bits (BUILD_ROAD) or 6 bits (BUILD_SETTLEMENT)

## BitBoard

We need to support the following operations:

- Action Generation:
  - Yielding Resources:
    - Need to tiles for a given number
    - Check if robber in given tile
    - Get buildings (color and multiplier) around tile
  - Road Building: Valid Buildable Edges
  - Settlement Posibilities: Buildable Node Ids
  - Robber Possibilities:
    - Which are land tiles not robber.
- Move Application
  - Build a settlement
  - Build a road
  - Build a city
  - Move Robber
- Queries
  - Adjacent Tiles to a Node (for yielding resources in initial building phase)
  - Edges given a Node Id
  - Is friendly road (edgeId, color)
  - Is Enemy Node (nodeId, color)
  - Expandable Nodes (for )

Additional Responsabilities

- Incrementally Mantain Longest Road (Color + Count)
- Buildable Subgraph for quick Answering Buildable Edges
- Connected Components (this is to, when plowing happens, be able to count by just max(len) pieces after updating data structure). Otherwise we would have to re-BFS all roads.
