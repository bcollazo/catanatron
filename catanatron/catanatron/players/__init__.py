"""Builtin players, and their registration in the global registry.

Importing this package is what makes the builtin bots addressable by key from
the CLI (``--players=R,AB:depth=3``) and from the web API.
"""

from catanatron.models.player import HumanPlayer, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.registry import REGISTRY

__all__ = [
    "AlphaBetaPlayer",
    "GreedyPlayoutsPlayer",
    "HumanPlayer",
    "MCTSPlayer",
    "RandomPlayer",
    "SameTurnAlphaBetaPlayer",
    "ValueFunctionPlayer",
    "VictoryPointPlayer",
    "WeightedRandomPlayer",
    "register_builtins",
]


def register_builtins(registry=REGISTRY):
    """Register the builtin players. Idempotent."""
    builtins = [
        ("H", HumanPlayer, "Human (terminal)"),
        ("R", RandomPlayer, "Random"),
        ("W", WeightedRandomPlayer, "Weighted Random"),
        ("VP", VictoryPointPlayer, "Victory Point"),
        ("F", ValueFunctionPlayer, "Value Function"),
        ("G", GreedyPlayoutsPlayer, "Greedy Playouts"),
        ("M", MCTSPlayer, "MCTS"),
        ("AB", AlphaBetaPlayer, "AlphaBeta"),
        ("SAB", SameTurnAlphaBetaPlayer, "Same-Turn AlphaBeta"),
    ]
    for key, player_class, name in builtins:
        registry.register(key, player_class, name=name, replace=True)
    return registry


register_builtins()
