import random
from enum import Enum
from collections import namedtuple, defaultdict

from catanatron.coordinate_system import (
    generate_coordinate_system,
    Direction,
    add,
    UNIT_VECTORS,
)


class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


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


class Tile:
    next_autoinc_id = 0

    def __init__(self, resource, number, nodes, edges):
        self.id = Tile.next_autoinc_id
        Tile.next_autoinc_id += 1

        self.resource = resource  # None means desert tile
        self.number = number
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Tile:" + str(self.resource)


class Port:
    next_autoinc_id = 0

    def __init__(self, resource, direction, nodes, edges):
        self.id = Port.next_autoinc_id
        Port.next_autoinc_id += 1

        self.resource = resource  # None means its a 3:1 port.
        self.direction = direction
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Port:" + str(self.resource)


class Water:
    next_autoinc_id = 0

    def __init__(self, nodes, edges):
        self.id = Water.next_autoinc_id
        Water.next_autoinc_id += 1

        self.nodes = nodes
        self.edges = edges


class Edge:
    next_autoinc_id = 0

    def __init__(self):
        self.id = Edge.next_autoinc_id
        Edge.next_autoinc_id += 1

    def __repr__(self):
        return "Edge:" + str(self.id)


class Node:
    next_autoinc_id = 0

    def __init__(self):
        self.id = Node.next_autoinc_id
        Node.next_autoinc_id += 1

    def __repr__(self):
        return "Node:" + str(self.id)


class BaseMap:
    """
    Describes a basic 4 player map. Includes the tiles, ports, and numbers used.
    """

    def __init__(self):
        self.numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        self.port_resources = [
            # These are 2:1 ports
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
            # These represet 3:1 ports
            None,
            None,
            None,
            None,
        ]
        self.tile_resources = [
            # Four wood tiles
            Resource.WOOD,
            Resource.WOOD,
            Resource.WOOD,
            Resource.WOOD,
            # Three brick tiles
            Resource.BRICK,
            Resource.BRICK,
            Resource.BRICK,
            # Four sheep tiles
            Resource.SHEEP,
            Resource.SHEEP,
            Resource.SHEEP,
            Resource.SHEEP,
            # Four wheat tiles
            Resource.WHEAT,
            Resource.WHEAT,
            Resource.WHEAT,
            Resource.WHEAT,
            # Three ore tiles
            Resource.ORE,
            Resource.ORE,
            Resource.ORE,
            # One desert
            None,
        ]

        # 3 layers, where last layer is water
        self.coordinate_system = generate_coordinate_system(3)
        self.topology = {
            # center
            (0, 0, 0): Tile,
            # first layer
            (1, -1, 0): Tile,
            (0, -1, 1): Tile,
            (-1, 0, 1): Tile,
            (-1, 1, 0): Tile,
            (0, 1, -1): Tile,
            (1, 0, -1): Tile,
            # second layer
            (2, -2, 0): Tile,
            (1, -2, 1): Tile,
            (0, -2, 2): Tile,
            (-1, -1, 2): Tile,
            (-2, 0, 2): Tile,
            (-2, 1, 1): Tile,
            (-2, 2, 0): Tile,
            (-1, 2, -1): Tile,
            (0, 2, -2): Tile,
            (1, 1, -2): Tile,
            (2, 0, -2): Tile,
            (2, -1, -1): Tile,
            # third (water) layer
            (3, -3, 0): (Port, Direction.WEST),
            (2, -3, 1): Water,
            (1, -3, 2): (Port, Direction.NORTHWEST),
            (0, -3, 3): Water,
            (-1, -2, 3): (Port, Direction.NORTHWEST),
            (-2, -1, 3): Water,
            (-3, 0, 3): (Port, Direction.NORTHEAST),
            (-3, 1, 2): Water,
            (-3, 2, 1): (Port, Direction.EAST),
            (-3, 3, 0): Water,
            (-2, 3, -1): (Port, Direction.EAST),
            (-1, 3, -2): Water,
            (0, 3, -3): (Port, Direction.SOUTHEAST),
            (1, 2, -3): Water,
            (2, 1, -3): (Port, Direction.SOUTHWEST),
            (3, 0, -3): Water,
            (3, -1, -2): (Port, Direction.SOUTHWEST),
            (3, -2, -1): Water,
        }


# TODO: Build "deck" of these (14 roads, 5 settlements, 4 cities)
class BuildingType(Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    ROAD = "ROAD"


Building = namedtuple("Building", ["color", "building_type"])


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
    # graph is { node => { edge: node }{<=3}}
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
        self.graph = graph  #  { node => { edge: node }{<=3}}

        # (coordinate, nodeRef | edgeRef) | node | edge => None | Building
        self.buildings = {}

        # Should we?
        # node + nodeedgeref must be able to query edge.
        # self.edges.get((node, nodeedgeref))
        # self.edges.get((coordinate, edgeref))
        # self.edges.get(edge_id)

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
        if not initial_placement:
            # TODO: Check connectedness
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        # we add and check in multiple representations to ease querying
        keys = [(coordinate, nodeRef), self.nodes[(coordinate, nodeRef)]]
        exists = map(lambda k: self.buildings.get(k) is not None, keys)
        if any(exists):
            raise ValueError("Invalid Settlement Placement: a building exists there")

        building = Building(color=color, building_type=BuildingType.SETTLEMENT)
        for key in keys:
            self.buildings[key] = building

    def build_road(self, color, coordinate, edgeRef):
        edge_under_consideration = self.edges.get((coordinate, edgeRef))

        a_noderef, b_noderef = get_edge_nodes(edgeRef)
        a_node = self.nodes.get((coordinate, a_noderef))
        b_node = self.nodes.get((coordinate, b_noderef))
        a_color = self.get_color(a_node)
        b_color = self.get_color(b_node)

        nothing_there = self.buildings.get(edge_under_consideration) is None
        one_end_has_color = self.is_color(a_node, color) or self.is_color(b_node, color)
        a_connected = any(
            [
                self.is_color(edge, color)
                for edge in self.graph.get(a_node).keys()
                if edge_under_consideration != edge
            ]
        )
        b_connected = any(
            [
                self.is_color(edge, color)
                for edge in self.graph.get(b_node).keys()
                if edge_under_consideration != edge
            ]
        )
        enemy_on_a = a_color is not None and a_color != color
        enemy_on_b = b_color is not None and b_color != color

        can_build = nothing_there and (
            one_end_has_color
            or (a_connected and not enemy_on_a)
            or (b_connected and not enemy_on_b)
        )
        if not can_build:
            raise ValueError("Invalid Road Placement: not connected")

        # we add and check in multiple representations to ease querying
        keys = [(coordinate, edgeRef), self.edges[(coordinate, edgeRef)]]
        exists = map(lambda k: self.buildings.get(k) is not None, keys)
        if any(exists):
            raise ValueError("Invalid Road Placement: a road exists there")

        building = Building(color=color, building_type=BuildingType.ROAD)
        for key in keys:
            self.buildings[key] = building

    def buildable_nodes(self, color, initial_placement=False):
        buildable = set()

        def is_buildable(node):
            # is buildable if this and neighboring nodes are empty
            # doesn't check for connected-ness
            under_consideration = [node] + list(self.graph[node].values())
            has_building = map(
                lambda n: self.buildings.get(n) is None,
                under_consideration,
            )

            return all(has_building)

        # if initial-placement, iterate over non-water/port tiles, for each
        # of these nodes check if its a buildable node.
        if initial_placement:
            for (coordinate, tile) in self.tiles.items():
                if isinstance(tile, Port) or isinstance(tile, Water):
                    continue

                for (noderef, node) in tile.nodes.items():
                    if is_buildable(node):
                        buildable.add(node)

        # if not initial-placement, find all connected components. For each
        #   node in this connected subgraph, iterate checking buildability
        connected_components = self.find_connected_components(color)
        for subgraph in connected_components:
            for node in subgraph.keys():
                # by definition node is "connected", so only need to check buildable
                if is_buildable(node):
                    buildable.add(node)

        return buildable

    # ===== Helper functions
    def get_color(self, building_key):
        """None if no one has built here, else builder's color"""
        building = self.buildings.get(building_key)
        return None if building is None else building.color

    def is_color(self, building_key, color):
        """boolean on whether this color has built here (edge or node)"""
        return self.get_color(building_key) == color

    def find_connected_components(self, color):
        """returns connected subgraphs for a given player

        algorithm goes like: find all nodes where color has buildings.
        start a BFS from any of these nodes, only following edges color owns,
        appending to subgraph and eliminating from agenda if builded there.
        repeat until list of settled_nodes is empty.

        Args:
            color (Color): [description]

        Returns:
            [list of self.graph-like objects]: connected subgraphs. subgraph
                will include nodes that color might not own, just to make it
                "closed" and easier for buildable_nodes to operate.
        """
        settled_nodes = set(
            node for node in self.nodes.values() if self.is_color(node, color)
        )
        settled_edges = set(
            edge for edge in self.edges.values() if self.is_color(edge, color)
        )
        # TODO: Include roads settled as well.
        # subgraph will be inclusing of bordering nodes (even if others have built there)
        # "buildable" will be all nodes here, as long as distance 2 of other buildings.
        subgraphs = []
        while len(settled_edges) > 0:
            tmp_subgraph = defaultdict(dict)

            # start bfs
            agenda = [settled_edges.pop()]
            visited = set()
            while len(agenda) > 0:
                edge = agenda.pop()
                visited.add(edge)
                if edge in settled_edges:
                    settled_edges.remove(edge)

                # can't imagine a better way to get the two "end" nodes
                # given a edge, _without_ having the coordinate tile.
                pair = None
                for node, neighbors_map in self.graph.items():
                    if edge in neighbors_map:
                        pair = (node, neighbors_map[edge])
                assert pair is not None

                # add to subgraph
                tmp_subgraph[pair[0]][edge] = pair[1]
                tmp_subgraph[pair[1]][edge] = pair[0]

                # edges to add to exploration are ones we are connected to.
                # TODO: This can prob get simplified:
                a_color = self.get_color(pair[0])
                if a_color is not None and a_color != color:  # enemy has a
                    a_candidates = []  # dont expand this way
                else:
                    a_candidates = [
                        candidate_edge
                        for candidate_edge, neighbor in self.graph[pair[0]].items()
                        if (
                            candidate_edge != edge
                            and self.is_color(candidate_edge, color)
                        )
                    ]
                b_color = self.get_color(pair[1])
                if b_color is not None and b_color != color:  # enemy has b
                    b_candidates = []  # dont expand this way
                else:
                    b_candidates = [
                        candidate_edge
                        for candidate_edge, neighbor in self.graph[pair[1]].items()
                        if (
                            candidate_edge != edge
                            and self.is_color(candidate_edge, color)
                        )
                    ]

                for candidate_edge in a_candidates + b_candidates:
                    if candidate_edge not in visited and candidate_edge not in agenda:
                        agenda.append(candidate_edge)

            subgraphs.append(dict(tmp_subgraph))
        return subgraphs


class Game:
    def __init__(self):
        # Map is defined by having a large-enough coordinate_system
        # and reading from a (coordinate) => Tile | Water | Port map
        # and filling the rest with water tiles.
        self.map = BaseMap()

        # board should be: (coordinate) => Tile (with nodes and edges initialized)
        self.board = Board(self.map)

        # TODO: Create players.


def get_nodes_and_edges(board, coordinate):
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
        if coord not in board:
            continue

        neighbor = board[coord]
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
            edges[edgeref] = Edge()

    return nodes, edges
