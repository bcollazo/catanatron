import random
from enum import Enum

import networkx as nx

from catanatron.models.coordinate_system import Direction, add, UNIT_VECTORS
from catanatron.models.map import Tile, Water, Port

# Given a tile, the reference to the node.
class NodeRef(Enum):
    NORTH = "NORTH"
    NORTHEAST = "NORTHEAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTH = "SOUTH"
    SOUTHWEST = "SOUTHWEST"
    NORTHWEST = "NORTHWEST"


# References an edge from a tile.
class EdgeRef(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


# TODO: Add typing information
def initialize_board(catan_map):
    shuffled_port_resources = random.sample(
        catan_map.port_resources, len(catan_map.port_resources)
    )
    shuffled_tile_resources = random.sample(
        catan_map.tile_resources, len(catan_map.tile_resources)
    )
    shuffled_numbers = random.sample(catan_map.numbers, len(catan_map.numbers))

    # for each topology entry, place a tile. keep track of nodes and edges
    all_tiles = {}
    node_autoinc = 0
    tile_autoinc = 0
    port_autoinc = 0
    nxgraph = nx.Graph()
    for (coordinate, tile_type) in catan_map.topology.items():
        nodes, edges, node_autoinc, nxgraph = get_nodes_and_edges(
            all_tiles, coordinate, node_autoinc, nxgraph
        )

        # create and save tile
        if isinstance(tile_type, tuple):  # is port
            (_, direction) = tile_type
            port = Port(
                port_autoinc, shuffled_port_resources.pop(), direction, nodes, edges
            )
            all_tiles[coordinate] = port
            port_autoinc += 1
        elif tile_type == Tile:
            resource = shuffled_tile_resources.pop()
            if resource != None:
                number = shuffled_numbers.pop()
                tile = Tile(tile_autoinc, resource, number, nodes, edges)
            else:
                tile = Tile(tile_autoinc, None, None, nodes, edges)  # desert
            all_tiles[coordinate] = tile
            tile_autoinc += 1
        elif tile_type == Water:
            water_tile = Water(nodes, edges)
            all_tiles[coordinate] = water_tile
        else:
            raise Exception("Something went wrong")

    return (all_tiles, nxgraph)


def get_nodes_and_edges(tiles, coordinate, node_autoinc, nxgraph):
    """Get pre-existing nodes and edges in board for given tile coordinate"""
    nodes = {
        NodeRef.NORTH: None,
        NodeRef.NORTHEAST: None,
        NodeRef.SOUTHEAST: None,
        NodeRef.SOUTH: None,
        NodeRef.SOUTHWEST: None,
        NodeRef.NORTHWEST: None,
    }
    edges = {
        EdgeRef.EAST: None,
        EdgeRef.SOUTHEAST: None,
        EdgeRef.SOUTHWEST: None,
        EdgeRef.WEST: None,
        EdgeRef.NORTHWEST: None,
        EdgeRef.NORTHEAST: None,
    }

    # Find pre-existing ones
    neighbor_tiles = [(add(coordinate, UNIT_VECTORS[d]), d) for d in Direction]
    for (coord, neighbor_direction) in neighbor_tiles:
        if coord not in tiles:
            continue

        neighbor = tiles[coord]
        if neighbor_direction == Direction.EAST:
            nodes[NodeRef.NORTHEAST] = neighbor.nodes[NodeRef.NORTHWEST]
            nodes[NodeRef.SOUTHEAST] = neighbor.nodes[NodeRef.SOUTHWEST]
            edges[EdgeRef.EAST] = neighbor.edges[EdgeRef.WEST]
        elif neighbor_direction == Direction.SOUTHEAST:
            nodes[NodeRef.SOUTH] = neighbor.nodes[NodeRef.NORTHWEST]
            nodes[NodeRef.SOUTHEAST] = neighbor.nodes[NodeRef.NORTH]
            edges[EdgeRef.SOUTHEAST] = neighbor.edges[EdgeRef.NORTHWEST]
        elif neighbor_direction == Direction.SOUTHWEST:
            nodes[NodeRef.SOUTH] = neighbor.nodes[NodeRef.NORTHEAST]
            nodes[NodeRef.SOUTHWEST] = neighbor.nodes[NodeRef.NORTH]
            edges[EdgeRef.SOUTHWEST] = neighbor.edges[EdgeRef.NORTHEAST]
        elif neighbor_direction == Direction.WEST:
            nodes[NodeRef.NORTHWEST] = neighbor.nodes[NodeRef.NORTHEAST]
            nodes[NodeRef.SOUTHWEST] = neighbor.nodes[NodeRef.SOUTHEAST]
            edges[EdgeRef.WEST] = neighbor.edges[EdgeRef.EAST]
        elif neighbor_direction == Direction.NORTHWEST:
            nodes[NodeRef.NORTH] = neighbor.nodes[NodeRef.SOUTHEAST]
            nodes[NodeRef.NORTHWEST] = neighbor.nodes[NodeRef.SOUTH]
            edges[EdgeRef.NORTHWEST] = neighbor.edges[EdgeRef.SOUTHEAST]
        elif neighbor_direction == Direction.NORTHEAST:
            nodes[NodeRef.NORTH] = neighbor.nodes[NodeRef.SOUTHWEST]
            nodes[NodeRef.NORTHEAST] = neighbor.nodes[NodeRef.SOUTH]
            edges[EdgeRef.NORTHEAST] = neighbor.edges[EdgeRef.SOUTHWEST]
        else:
            raise Exception("Something went wrong")

    # Initializes new ones
    for noderef, value in nodes.items():
        if value is None:
            nodes[noderef] = node_autoinc
            node_autoinc += 1
            nxgraph.add_node(node_autoinc)
    for edgeref, value in edges.items():
        if value is None:
            a_noderef, b_noderef = get_edge_nodes(edgeref)
            edge_nodes = (nodes[a_noderef], nodes[b_noderef])
            edges[edgeref] = edge_nodes
            nxgraph.add_edge(*edge_nodes)

    return nodes, edges, node_autoinc, nxgraph


def get_edge_nodes(edge_ref):
    """returns pair of nodes at the "ends" of a given edge"""
    return {
        EdgeRef.EAST: (NodeRef.NORTHEAST, NodeRef.SOUTHEAST),
        EdgeRef.SOUTHEAST: (NodeRef.SOUTHEAST, NodeRef.SOUTH),
        EdgeRef.SOUTHWEST: (NodeRef.SOUTH, NodeRef.SOUTHWEST),
        EdgeRef.WEST: (NodeRef.SOUTHWEST, NodeRef.NORTHWEST),
        EdgeRef.NORTHWEST: (NodeRef.NORTHWEST, NodeRef.NORTH),
        EdgeRef.NORTHEAST: (NodeRef.NORTH, NodeRef.NORTHEAST),
    }[edge_ref]


# TODO: Could consolidate Direction with EdgeRef.
PORT_DIRECTION_TO_NODEREFS = {
    Direction.WEST: (NodeRef.NORTHWEST, NodeRef.SOUTHWEST),
    Direction.NORTHWEST: (NodeRef.NORTH, NodeRef.NORTHWEST),
    Direction.NORTHEAST: (NodeRef.NORTHEAST, NodeRef.NORTH),
    Direction.EAST: (NodeRef.SOUTHEAST, NodeRef.NORTHEAST),
    Direction.SOUTHEAST: (NodeRef.SOUTH, NodeRef.SOUTHEAST),
    Direction.SOUTHWEST: (NodeRef.SOUTHWEST, NodeRef.SOUTH),
}
