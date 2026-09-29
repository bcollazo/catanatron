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
from typing import Dict, FrozenSet, Optional, Tuple

from catanatron.models.coordinate_system import Coordinate
from catanatron.models.enums import FastBuildingType, FastResource
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
class PublicMap:
    """Pure-data snapshot of the static board layout.

    The CatanMap (tiles, numbers, ports, terrain nodes) is fixed at game start,
    so it is snapshotted once per decision rather than changing turn to turn.
    Keyed absolutely by tile/port/node id, so agents can reason about where
    things are instead of only parsing the flat ``TILE*``/``PORT*`` features.
    """

    tiles: Dict[int, Tuple[Optional[FastResource], Optional[int]]]
    """tile_id -> (resource, roll); desert is (None, None)."""
    tile_coordinates: Dict[int, Coordinate]
    """tile_id -> its cube coordinate; bridges id-keyed tiles to coordinate-keyed actions."""
    ports: Dict[int, Tuple[Optional[FastResource], Tuple[NodeId, NodeId]]]
    """port_id -> (resource, (node_a, node_b)) trading nodes; resource None means 3:1."""
    adjacent_tiles: Dict[NodeId, Tuple[int, ...]]
    """node_id -> tile ids touching it; edge to per-node production."""
    land_nodes: FrozenSet[NodeId]
    """All node ids on land (where settlements may legally be built)."""


@dataclass(frozen=True)
class PublicBoard:
    """Publicly knowable facts about the shared board."""

    buildings: Dict[NodeId, Tuple[Color, FastBuildingType]]
    roads: Dict[Tuple[NodeId, NodeId], Color]
    robber_tile_id: int
    """Id of the tile the robber sits on, resolvable to/from a coordinate via ``map.tile_coordinates``."""
    longest_road_color: Optional[Color]
    longest_road_length: int
    map: PublicMap
    """Static terrain; see ``PublicMap``."""


@dataclass(frozen=True)
class PublicState:
    """Pure-data snapshot of every public fact in a State."""

    board: PublicBoard
    players: Dict[Color, PublicPlayer]
