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
    builtins = {
        "H": HumanPlayer,
        "R": RandomPlayer,
        "W": WeightedRandomPlayer,
        "VP": VictoryPointPlayer,
        "F": ValueFunctionPlayer,
        "G": GreedyPlayoutsPlayer,
        "M": MCTSPlayer,
        "AB": AlphaBetaPlayer,
        "SAB": SameTurnAlphaBetaPlayer,
    }
    for key, player_class in builtins.items():
        registry.register(key, player_class, replace=True)
    return registry


register_builtins()
