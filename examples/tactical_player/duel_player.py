"""Native CLI entry for the frozen two-player challenge policy."""

from pathlib import Path

from catanatron.sources import load_module


HERE = Path(__file__).resolve().parent
policy = load_module(str(HERE / "agent.py"))


class TacticalDuelPlayer(policy.CatanAgent):
    LABEL = "Tactical Duel"

    def __init__(self, color, params=None):
        super().__init__(color, policy.Weights.load(HERE / "weights-3p-tactical.json"), negotiate=False)
