---
icon: webhook
---

# Simulation Hooks

Anything that watches a game implements the same three hooks. Players use
them too — see [Custom Bots](../advanced/editor.md) — so there is one
lifecycle to learn:

| hook | when |
|------|------|
| `before(game)` | once, before any action. The board is decided. |
| `step(game_before_action, action)` | every action taken by anyone. |
| `after(game)` | once the game is over. |

An **accumulator** is an observer that only watches. Write a file like
`mycode.py`:

```python
from catanatron import ActionType
from catanatron.cli import SimulationAccumulator

class PortTradeCounter(SimulationAccumulator):
    def before_all(self):
        self.num_trades = 0

    def step(self, game_before_action, action):
        if action.action_type == ActionType.MARITIME_TRADE:
            self.num_trades += 1

    def after_all(self):
        print(f'There were {self.num_trades} trades with the bank!')
```

Then point `--accumulator` at it:

```bash
catanatron-play --accumulator mycode.py --players=R,R
```

`--accumulator` is repeatable, and takes the same kind of source as `--bot`:
a file or an importable module, optionally naming a class after a `#`.

An accumulator holds the live `game`, so it sees everything. To serialize what
it sees, pick the flavor the job needs:

```python
from catanatron.serialization import client_view, state_to_json

doc = state_to_json(game)
client_view(doc)                 # a spectator: every hand face up
client_view(doc, game.state.current_color())   # only what that seat knows
```

Collecting training data usually wants the first; recording what a player
could actually have reasoned from wants the second.

`SimulationAccumulator` adds two hooks on top of the three above, for the whole
batch rather than a single game:

- `before_all()` — before the first game
- `after_all()` — after the last game
