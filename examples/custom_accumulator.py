from catanatron import ActionType
from catanatron.cli import SimulationAccumulator, register_cli_accumulator


class PortTradeCounter(SimulationAccumulator):
    def before_all(self):
        self.num_trades = 0

    def step(self, game_before_action, action):
        if action.action_type == ActionType.MARITIME_TRADE:
            self.num_trades += 1

    def after_all(self):
        print(f"There were {self.num_trades} port trades!")


register_cli_accumulator(PortTradeCounter)
