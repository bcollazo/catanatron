"""
Typed, pure-data snapshot of everything publicly observable in a game.

These dataclasses mirror the engine's public board and per-player public
counts, keyed absolutely (node/edge ids and Color), unlike ``features``
which is keyed relative to the observing color. They are built by the engine
adapter (``_build_public_state`` in ``perspective_player``) and carried on
``Observation.public_state``, so fair agents get structured access without any
Game/State reference. Opponent hand identities and actual victory points are
never included.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from catanatron.models.enums import FastBuildingType
from catanatron.models.map import NodeId
from catanatron.models.player import Color


@dataclass(frozen=True)
class PublicPlayer:
    """Publicly knowable facts about one color's position."""

    public_vps: int
    has_army: bool
    has_road: bool
    longest_road_length: int
    roads_left: int
    settlements_left: int
    cities_left: int
    has_rolled: bool
    hand_resource_count: int
    hand_dev_count: int
    played_knight: int
    played_monopoly: int
    played_road_building: int
    played_year_of_plenty: int
    played_victory_point: int


@dataclass(frozen=True)
class PublicBoard:
    """Publicly knowable facts about the shared board."""

    buildings: Dict[NodeId, Tuple[Color, FastBuildingType]]
    roads: Dict[Tuple[NodeId, NodeId], Color]
    robber_coordinate: Tuple[int, int, int]
    longest_road_color: Optional[Color]
    longest_road_length: int


@dataclass(frozen=True)
class PublicState:
    """Pure-data snapshot of every public fact in a State."""

    board: PublicBoard
    players: Dict[Color, PublicPlayer]
