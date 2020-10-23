from enum import Enum
from collections import namedtuple, defaultdict
from typing import Dict

from catanatron.models.player import Color
from catanatron.models.map import BaseMap, Water, Port
from catanatron.models.board_initializer import (
    initialize_board,
    Node,
    Edge,
    PORT_DIRECTION_TO_NODEREFS,
)


# TODO: Build "deck" of these (14 roads, 5 settlements, 4 cities)
class BuildingType(Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    ROAD = "ROAD"


Building = namedtuple("Building", ["color", "building_type"])


Graph = Dict[Node, Dict[Edge, Node]]


class Board:
    """Tries to encapsulate all state information regarding the board"""

    def __init__(self, players=[c for c in Color], catan_map=None):
        """
        Initializes a new random board, based on the catan_map description.
        It first shuffles tiles, ports, and numbers. Then goes satisfying the
        topology (placing tiles on coordinates); ensuring to "attach" these to
        neighbor tiles. (no repeated nodes or edges objects).
        """
        self.map = catan_map or BaseMap()

        tiles, nodes, edges, graph = initialize_board(self.map)
        self.tiles = tiles  # (coordinate) => Tile (with nodes and edges initialized)
        self.nodes = nodes  # (coordinate, noderef) => node
        self.edges = edges  # (coordinate, edgeref) => edge
        self.graph = graph  #  { node => { edge: node }}
        self.buildings = {}  #  node | edge => None | Building

        # assumes there is at least one desert:
        self.robber_coordinate = filter(
            lambda coordinate: tiles[coordinate].resource == None, tiles.keys()
        ).__next__()

    def build_settlement(self, color, node, initial_build_phase=False):
        """Adds a settlement, and ensures is a valid place to build.

        Args:
            color (Color): player's color
            node (Node): where to build
            initial_build_phase (bool, optional):
                Whether this is part of initial building phase, so as to skip
                connectedness validation. Defaults to True.
        """
        buildable = self.buildable_nodes(color, initial_build_phase=initial_build_phase)
        if node not in buildable:
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        if self.buildings.get(node) is not None:
            raise ValueError("Invalid Settlement Placement: a building exists there")

        building = Building(color=color, building_type=BuildingType.SETTLEMENT)
        self.buildings[node] = building
        return building

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        if edge not in buildable:
            raise ValueError("Invalid Road Placement: not connected")

        if self.buildings.get(edge) is not None:
            raise ValueError("Invalid Road Placement: a road exists there")

        building = Building(color=color, building_type=BuildingType.ROAD)
        self.buildings[edge] = building
        return building

    def build_city(self, color, node):
        building = self.buildings.get(node)
        if (
            building is None
            or building.color != color
            or building.building_type != BuildingType.SETTLEMENT
        ):
            raise ValueError("Invalid City Placement: no player settlement there")

        building = Building(color=color, building_type=BuildingType.CITY)
        self.buildings[node] = building
        return building

    def buildable_nodes(self, color: Color, initial_build_phase=False):
        buildable = set()

        def is_buildable(node):
            """true if this and neighboring nodes are empty"""
            under_consideration = [node] + list(self.graph[node].values())
            has_building = map(
                lambda n: self.buildings.get(n) is None,
                under_consideration,
            )
            return all(has_building)

        # if initial-placement, iterate over non-water/port tiles, for each
        # of these nodes check if its a buildable node.
        if initial_build_phase:
            for (coordinate, tile) in self.resource_tiles():
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

    def buildable_edges(self, color: Color):
        # A water-edge is an edge where both adjacent tiles are water/ports
        def is_buildable(edge):  # assumes not a water-edge
            a_node, b_node = edge.nodes
            a_color = self.get_color(a_node)
            b_color = self.get_color(b_node)

            # check if buildable. buildable if nothing there, connected (one end_has_color)
            nothing_there = self.buildings.get(edge) is None
            one_end_has_color = self.is_color(a_node, color) or self.is_color(
                b_node, color
            )
            a_connected = any(
                [
                    self.is_color(e, color)
                    for e in self.graph.get(a_node).keys()
                    if e != edge
                ]
            )
            b_connected = any(
                [
                    self.is_color(e, color)
                    for e in self.graph.get(b_node).keys()
                    if e != edge
                ]
            )
            enemy_on_a = a_color is not None and a_color != color
            enemy_on_b = b_color is not None and b_color != color

            can_build = nothing_there and (
                one_end_has_color  # helpful for initial_build_phase
                or (a_connected and not enemy_on_a)
                or (b_connected and not enemy_on_b)
            )
            return can_build

        buildable = set()

        for (coordinate, tile) in self.resource_tiles():
            for (edgeref, edge) in tile.edges.items():
                if is_buildable(edge):
                    buildable.add(edge)

        return buildable

    def resource_tiles(self):
        for (coordinate, tile) in self.tiles.items():
            if isinstance(tile, Port) or isinstance(tile, Water):
                continue
            yield (coordinate, tile)

    def get_adjacent_tiles(self, node_or_edge):
        if isinstance(node_or_edge, Node):
            for coordinate, tile in self.resource_tiles():
                if node_or_edge in tile.nodes.values():
                    yield tile
        else:
            for coordinate, tile in self.resource_tiles():
                if node_or_edge in tile.edges.values():
                    yield tile

    def get_player_buildings(self, color, building_type=None):
        buildings = filter(
            lambda x: x[1].color == color,
            self.buildings.items(),
        )  # (node, building) sequence
        if building_type is not None:
            buildings = filter(lambda x: x[1].building_type == building_type, buildings)

        return list(buildings)

    def get_port_nodes(self):
        """Yields (node, resource) tuples"""
        for (coordinate, value) in self.map.topology.items():
            if not isinstance(value, tuple):
                continue

            _, direction = value
            (a_noderef, b_noderef) = PORT_DIRECTION_TO_NODEREFS[direction]
            yield (self.nodes[(coordinate, a_noderef)], self.tiles[coordinate].resource)
            yield (self.nodes[(coordinate, b_noderef)], self.tiles[coordinate].resource)

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        ports = []
        for node, resource in self.get_port_nodes():
            building = self.buildings.get(node)
            if building is not None and building.color == color:
                yield resource

    def find_connected_components(self, color: Color):
        """returns connected subgraphs for a given player

        algorithm goes like: find all nodes where color has buildings.
        start a BFS from any of these nodes, only following edges color owns,
        appending to subgraph and eliminating from agenda if builded there.
        repeat until list of settled_nodes is empty.

        Returns:
            [list of self.graph-like objects]: connected subgraphs. subgraph
                might include nodes that color doesnt own (on the way and on ends),
                just to make it is "closed" and easier for buildable_nodes to operate.
        """
        settled_edges = set(
            edge for edge in self.edges.values() if self.is_color(edge, color)
        )

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

                # add to subgraph
                tmp_subgraph[edge.nodes[0]][edge] = edge.nodes[1]
                tmp_subgraph[edge.nodes[1]][edge] = edge.nodes[0]

                # edges to add to exploration are ones we are connected to.
                candidates = set()  # will be explorable "5-edge star" around edge
                a_color = self.get_color(edge.nodes[0])
                if a_color is None or a_color == color:  # enemy is not blocking
                    for candidate_edge in self.graph[edge.nodes[0]].keys():
                        candidates.add(candidate_edge)
                b_color = self.get_color(edge.nodes[1])
                if b_color is None or b_color == color:  # enemy is not blocking
                    for candidate_edge in self.graph[edge.nodes[1]].keys():
                        candidates.add(candidate_edge)

                for candidate_edge in candidates:
                    if (
                        candidate_edge not in visited
                        and candidate_edge not in agenda
                        and candidate_edge != edge
                        and self.is_color(candidate_edge, color)
                    ):
                        agenda.append(candidate_edge)

            subgraphs.append(dict(tmp_subgraph))
        return subgraphs

    # ===== Helper functions
    def get_color(self, node_or_edge):
        """None if no one has built here, else builder's color"""
        building = self.buildings.get(node_or_edge)
        return None if building is None else building.color

    def is_color(self, node_or_edge, color):
        """boolean on whether this color has built here (edge or node)"""
        return self.get_color(node_or_edge) == color
