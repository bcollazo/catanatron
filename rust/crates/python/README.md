# Catanatron Rust batch extension

This optional package leaves the existing Python package metadata untouched. Build it with
`maturin build --release` from this directory, then install the resulting wheel.

```python
from catanatron_rust import Batch

batch = Batch(64, players=4, map="BASE", seed=7)
view = batch.observe_many(list(range(64)))
first_action = int(view["action_ids"][view["menu_offsets"][0]])
step = batch.step_many([0], [first_action])

# Terminal environments stay terminal until explicitly selected for reset.
done = step["terminal"] | step["truncated"]
batch.reset_many([i for i, value in enumerate(done) if value], [1000 + i for i, value in enumerate(done) if value])
```

`features` is an owned contiguous `int16[N, 194]` array using observation schema v1.
Menus are ragged: environment `i` uses `action_ids[menu_offsets[i]:menu_offsets[i+1]]`.
These dynamic IDs are valid for one decision generation only. Rewards are `int8[N, players]`:
`+1` for the winner, `-1` for other players, and zero before terminal or on truncation.
