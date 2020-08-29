import random
from enum import Enum
from collections import namedtuple, defaultdict

from catanatron.models.coordinate_system import Direction, add, UNIT_VECTORS
from catanatron.models.map import BaseMap, Tile, Water, Port
from catanatron.models.board_algorithms import buildable_nodes, buildable_edges


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


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


def get_edge_nodes(edgeRef):
    """returns pair of nodes at the "ends" of a given edge"""
    return {
        EdgeRef.EAST: (NodeRef.NORTHEAST, NodeRef.SOUTHEAST),
        EdgeRef.SOUTHEAST: (NodeRef.SOUTHEAST, NodeRef.SOUTH),
        EdgeRef.SOUTHWEST: (NodeRef.SOUTH, NodeRef.SOUTHWEST),
        EdgeRef.WEST: (NodeRef.SOUTHWEST, NodeRef.NORTHWEST),
        EdgeRef.NORTHWEST: (NodeRef.NORTHWEST, NodeRef.NORTH),
        EdgeRef.NORTHEAST: (NodeRef.NORTH, NodeRef.NORTHEAST),
    }[edgeRef]


class Edge:
    next_autoinc_id = 0

    def __init__(self, nodes):
        self.id = Edge.next_autoinc_id
        Edge.next_autoinc_id += 1

        self.nodes = nodes  # the 2 nodes at the ends

    def __repr__(self):
        return "Edge:" + str(self.id)


class Node:
    next_autoinc_id = 0

    def __init__(self):
        self.id = Node.next_autoinc_id
        Node.next_autoinc_id += 1

    def __repr__(self):
        return "Node:" + str(self.id)


# TODO: Build "deck" of these (14 roads, 5 settlements, 4 cities)
class BuildingType(Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    ROAD = "ROAD"


Building = namedtuple("Building", ["color", "building_type"])


class Board(dict):
    """Since rep is basically a dict of (coordinate) => Tile, we inhert dict"""

    def __init__(self, catan_map=None):
        """
        Initializes a new random board, based on the catan_map description.
        It first shuffles tiles, ports, and numbers. Then goes satisfying the
        topology (placing tiles on coordinates); ensuring to "attach" these to
        neighbor tiles. (no repeated nodes or edges objects)
        """
        catan_map = catan_map or BaseMap()
        tiles, nodes, edges, graph = initialize_board(catan_map)

        self.tiles = tiles  # (coordinate) => Tile (with nodes and edges initialized)
        self.nodes = nodes  # (coordinate, noderef) => node
        self.edges = edges  # (coordinate, edgeref) => edge
        self.graph = graph  #  { node => { edge: node }}

        # (coordinate, nodeRef | edgeRef) | node | edge => None | Building
        self.buildings = {}

    def build_settlement(self, color, coordinate, nodeRef, initial_placement=False):
        """Adds a settlement, and ensures is a valid place to build.

        Args:
            color (Color): player's color
            coordinate (tuple): (x,y,z) of tile
            nodeRef (NodeRef): which of the 6 nodes of given tile
            initial_placement (bool, optional):
                Whether this is part of initial building phase, so as to skip
                connectedness validation. Defaults to True.
        """
        buildable = buildable_nodes(self, color, initial_placement=initial_placement)
        node = self.nodes.get((coordinate, nodeRef))
        if node not in buildable:
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        # we add and check in multiple representations to ease querying
        keys = [(coordinate, nodeRef), node]
        exists = map(lambda k: self.buildings.get(k) is not None, keys)
        if any(exists):
            raise ValueError("Invalid Settlement Placement: a building exists there")

        building = Building(color=color, building_type=BuildingType.SETTLEMENT)
        for key in keys:
            self.buildings[key] = building

    def build_road(self, color, coordinate, edgeRef):
        buildable = buildable_edges(self, color)
        edge = self.edges.get((coordinate, edgeRef))
        if edge not in buildable:
            raise ValueError("Invalid Road Placement: not connected")

        # we add and check in multiple representations to ease querying
        keys = [(coordinate, edgeRef), edge]
        exists = map(lambda k: self.buildings.get(k) is not None, keys)
        if any(exists):
            raise ValueError("Invalid Road Placement: a road exists there")

        building = Building(color=color, building_type=BuildingType.ROAD)
        for key in keys:
            self.buildings[key] = building

    # ===== Helper functions
    def get_color(self, building_key):
        """None if no one has built here, else builder's color"""
        building = self.buildings.get(building_key)
        return None if building is None else building.color

    def is_color(self, building_key, color):
        """boolean on whether this color has built here (edge or node)"""
        return self.get_color(building_key) == color


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
    all_nodes = {}
    all_edges = {}
    # graph is { node => { edge: node }}
    graph = defaultdict(dict)
    for (coordinate, tile_type) in catan_map.topology.items():
        nodes, edges = get_nodes_and_edges(all_tiles, coordinate)

        # create and save tile
        if isinstance(tile_type, tuple):  # is port
            (TileClass, direction) = tile_type
            port = TileClass(shuffled_port_resources.pop(), direction, nodes, edges)
            all_tiles[coordinate] = port
        elif tile_type == Tile:
            resource = shuffled_tile_resources.pop()
            if resource != None:
                number = shuffled_numbers.pop()
                tile = Tile(resource, number, nodes, edges)
            else:
                tile = Tile(None, None, nodes, edges)  # desert
            all_tiles[coordinate] = tile
        elif tile_type == Water:
            water_tile = Water(nodes, edges)
            all_tiles[coordinate] = water_tile
        else:
            raise Exception("Something went wrong")

        # upsert keys => nodes for querying later
        for noderef, node in nodes.items():
            all_nodes[(coordinate, noderef)] = node
        for edgeref, edge in edges.items():
            all_edges[(coordinate, edgeref)] = edge

        # upsert connections in graph (bi-directional)
        # clock-wise
        graph[nodes[NodeRef.NORTH]][edges[EdgeRef.NORTHEAST]] = nodes[NodeRef.NORTHEAST]
        graph[nodes[NodeRef.NORTHEAST]][edges[EdgeRef.EAST]] = nodes[NodeRef.SOUTHEAST]
        graph[nodes[NodeRef.SOUTHEAST]][edges[EdgeRef.SOUTHEAST]] = nodes[NodeRef.SOUTH]
        graph[nodes[NodeRef.SOUTH]][edges[EdgeRef.SOUTHWEST]] = nodes[NodeRef.SOUTHWEST]
        graph[nodes[NodeRef.SOUTHWEST]][edges[EdgeRef.WEST]] = nodes[NodeRef.NORTHWEST]
        graph[nodes[NodeRef.NORTHWEST]][edges[EdgeRef.NORTHWEST]] = nodes[NodeRef.NORTH]

        # counter-clockwise
        graph[nodes[NodeRef.NORTH]][edges[EdgeRef.NORTHWEST]] = nodes[NodeRef.NORTHWEST]
        graph[nodes[NodeRef.NORTHWEST]][edges[EdgeRef.WEST]] = nodes[NodeRef.SOUTHWEST]
        graph[nodes[NodeRef.SOUTHWEST]][edges[EdgeRef.SOUTHWEST]] = nodes[NodeRef.SOUTH]
        graph[nodes[NodeRef.SOUTH]][edges[EdgeRef.SOUTHEAST]] = nodes[NodeRef.SOUTHEAST]
        graph[nodes[NodeRef.SOUTHEAST]][edges[EdgeRef.EAST]] = nodes[NodeRef.NORTHEAST]
        graph[nodes[NodeRef.NORTHEAST]][edges[EdgeRef.NORTHEAST]] = nodes[NodeRef.NORTH]

    return (all_tiles, all_nodes, all_edges, graph)


def get_nodes_and_edges(tiles, coordinate):
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
        if value == None:
            nodes[noderef] = Node()
    for edgeref, value in edges.items():
        if value == None:
            a_noderef, b_noderef = get_edge_nodes(edgeref)
            edge_nodes = (nodes[a_noderef], nodes[b_noderef])
            edges[edgeref] = Edge(edge_nodes)

    return nodes, edges
