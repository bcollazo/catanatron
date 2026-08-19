"""
Typed, pure-data snapshot of the observing player's own private hand.

Unlike ``public_state`` — which is restricted to facts every player can know —
this is the observer's full inventory: exact resource counts and the identity
of every development card held. It is built for the observer's color only and
carried on ``Observation.inventory``; it is never computed for opponents, whose
hands stay visible only as public counts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Inventory:
    """The observer's private hand, keyed by card name (lowercase)."""

    wood: int = 0
    brick: int = 0
    sheep: int = 0
    wheat: int = 0
    ore: int = 0
    knight: int = 0
    year_of_plenty: int = 0
    monopoly: int = 0
    road_building: int = 0
    victory_point: int = 0
    has_played_development_card: bool = False
