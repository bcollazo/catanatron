"""Players the web server can seat.

The web UI addresses bots by name ("CATANATRON"), so those names are
registered here as first-class registry keys. Only builtin, in-process
players are registered: the server never imports player code named in a
request.
"""

from dataclasses import dataclass

from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.registry import REGISTRY


class Catanatron(AlphaBetaPlayer):
    """Catanatron, the strongest builtin bot."""

    @dataclass(frozen=True)
    class Params(AlphaBetaPlayer.Params):
        depth: int = 2
        prunning: bool = True


class WebHumanPlayer(ValueFunctionPlayer):
    """A seat played by a human through the web UI.

    Decisions arrive as posted actions rather than from ``decide``; the
    inherited value function only acts as a fallback.
    """

    IS_BOT = False
    LABEL = "Human"


def register_web_players(registry=REGISTRY):
    """Register the web-facing player keys. Idempotent.

    Only two: a human seat, and Catanatron itself (AlphaBeta with prunning,
    which is a distinct configuration and the product's name for it). The
    other builtins are already registered under their own keys, so aliasing
    them here would just show duplicates in the UI's dropdown.
    """
    registry.register("CATANATRON", Catanatron, replace=True)
    registry.register("HUMAN", WebHumanPlayer, replace=True)
    return registry
