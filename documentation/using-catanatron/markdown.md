---
icon: webhook
---

# Simulation Hooks

The `Accumulator` class allows you to hook into important events during simulations.

For example, write a file like `mycode.py` and have:

```python
from catanatron import ActionType
from catanatron.cli import SimulationAccumulator, register_cli_accumulator

class PortTradeCounter(SimulationAccumulator):
  def before_all(self):
    self.num_trades = 0

  def step(self, game_before_action, action):
    if action.action_type == ActionType.MARITIME_TRADE:
      self.num_trades += 1

  def after_all(self):
    print(f'There were {self.num_trades} trades with the bank!')

register_cli_accumulator(PortTradeCounter)
```

Then `catanatron-play --code=mycode.py` will count the number of trades in all simulations.
