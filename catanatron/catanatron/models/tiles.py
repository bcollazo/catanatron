from dataclasses import dataclass
from typing import Dict, Tuple, Union

from catanatron.models.enums import FastResource, EdgeRef, NodeRef
from catanatron.models.coordinate_system import Direction

EdgeId = Tuple[int, int]
NodeId = int


@dataclass
class LandTile:
    id: int
    resource: Union[FastResource, None]  # None means desert tile
    number: Union[int, None]  # None if desert
    nodes: Dict[NodeRef, NodeId]  # node_ref => node_id
    edges: Dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self):
        return self.id


@dataclass
class Port:
    id: int
    resource: Union[FastResource, None]  # None means desert tile
    direction: Direction
    nodes: Dict[NodeRef, NodeId]  # node_ref => node_id
    edges: Dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self):
        return self.id


@dataclass(frozen=True)
class Water:
    nodes: Dict[NodeRef, int]
    edges: Dict[EdgeRef, EdgeId]


Tile = Union[LandTile, Port, Water]
