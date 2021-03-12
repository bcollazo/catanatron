from collections import defaultdict

import networkx as nx

from catanatron.models.player import Color
from catanatron.models.map import BaseMap, NUM_NODES
from catanatron.models.enums import BuildingType


NODE_DISTANCES = None
EDGES = None

# Used to find relationships between nodes and edges
sample_map = BaseMap()
STATIC_GRAPH = nx.Graph()
for tile in sample_map.tiles.values():
    STATIC_GRAPH.add_nodes_from(tile.nodes.values())
    STATIC_GRAPH.add_edges_from(tile.edges.values())


def get_node_distances():
    global NODE_DISTANCES, STATIC_GRAPH
    if NODE_DISTANCES is None:
        NODE_DISTANCES = nx.floyd_warshall(STATIC_GRAPH)

    return NODE_DISTANCES


def get_edges():
    global EDGES, STATIC_GRAPH
    if EDGES is None:
        EDGES = STATIC_GRAPH.subgraph(range(NUM_NODES)).edges()
    return EDGES


class Board:
    """Tries to encapsulate all state information regarding the board"""

    def __init__(self, catan_map=None, initialize=True):
        """
        Initializes a new random board, based on the catan_map description.
        It first shuffles tiles, ports, and numbers. Then goes satisfying the
        topology (placing tiles on coordinates); ensuring to "attach" these to
        neighbor tiles. (no repeated nodes or edges objects).
        """
        if initialize:
            self.map = catan_map or BaseMap()  # Static State (no need to copy)

            self.buildings = dict()  # node_id => (color, building_type)
            self.roads = dict()  # (node_id, node_id) => color

            # color => int{}[] (list of node_id sets) one per component
            #   nodes in sets are incidental (might not be owned by player)
            self.connected_components = defaultdict(list)

            # assumes there is at least one desert:
            self.robber_coordinate = filter(
                lambda coordinate: self.map.tiles[coordinate].resource is None,
                self.map.tiles.keys(),
            ).__next__()

    def build_settlement(self, color, node_id, initial_build_phase=False):
        """Adds a settlement, and ensures is a valid place to build.

        Args:
            color (Color): player's color
            node_id (int): where to build
            initial_build_phase (bool, optional):
                Whether this is part of initial building phase, so as to skip
                connectedness validation. Defaults to True.
        """
        buildable = self.buildable_node_ids(
            color, initial_build_phase=initial_build_phase
        )
        if node_id not in buildable:
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        if node_id in self.buildings:
            raise ValueError("Invalid Settlement Placement: a building exists there")

        self.buildings[node_id] = (color, BuildingType.SETTLEMENT)
        self.update_connected_components()

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        inverted_edge = (edge[1], edge[0])
        if edge not in buildable and inverted_edge not in buildable:
            raise ValueError("Invalid Road Placement: not connected")

        if self.get_edge_color(edge) is not None:
            raise ValueError("Invalid Road Placement: a road exists there")

        normalized_edge = tuple(sorted(edge))
        self.roads[normalized_edge] = color
        self.update_connected_components()

        # # Update connected components (add to right list, add to )
        # # Check incident nodes are not enemy. If any is enemy, nothing to merge.
        # # Can I: g = nx.Graph(), g.add_edges(player_edges), remove all enemy-nodes
        # #   check if connected components is less than?
        # # IDEA: when enemy builds, can mutate nx.Graph to change edge id. (hacky)
        # a_graph = (
        #     None
        #     if self.is_enemy_node(edge[0], color)
        #     else self.color_node_to_subgraphs[color].get(edge[0], None)
        # )
        # b_graph = (
        #     None
        #     if self.is_enemy_node(edge[1], color)
        #     else self.color_node_to_subgraphs[color].get(edge[1], None)
        # )
        # if a_graph is not None and b_graph is not None and a_graph != b_graph:
        #     # merge subgraphs into one (i.e. player 'connected' roads)
        #     self.connected_components[color].remove(a_graph)
        #     self.connected_components[color].remove(b_graph)
        #     c_graph = nx.Graph()
        #     c_graph.add_edges_from(a_graph.edges)
        #     c_graph.add_edges_from(b_graph.edges)
        #     c_graph.add_edge(*edge)
        #     self.connected_components[color].append(c_graph)
        #     for node_id in c_graph.nodes:
        #         self.color_node_to_subgraphs[color][node_id] = c_graph
        #         assert len(list(nx.connected_components(c_graph))) == 1
        # elif a_graph is not None and b_graph is not None:  # but a == b
        #     # player connected same subgraph (no need to add "other" node)
        #     self.color_node_to_subgraphs[color][edge[0]].add_edge(*edge)
        # elif a_graph is not None:  # but b_graph is None
        #     self.color_node_to_subgraphs[color][edge[0]].add_edge(*edge)
        #     self.color_node_to_subgraphs[color][edge[1]] = self.color_node_to_subgraphs[
        #         color
        #     ][edge[0]]
        # else:  # must be a b_graph edge, a_graph is None
        #     self.color_node_to_subgraphs[color][edge[1]].add_edge(*edge)
        #     self.color_node_to_subgraphs[color][edge[0]] = self.color_node_to_subgraphs[
        #         color
        #     ][edge[1]]

    def build_city(self, color, node_id):
        building = self.buildings.get(node_id, None)
        if (
            building is None
            or building[0] != color
            or building[1] != BuildingType.SETTLEMENT
        ):
            raise ValueError("Invalid City Placement: no player settlement there")

        self.buildings[node_id] = (color, BuildingType.CITY)

    def buildable_node_ids(self, color: Color, initial_build_phase=False):
        buildable = set()

        def is_buildable(node_id):
            """true if this and neighboring nodes are empty"""
            under_consideration = [node_id] + list(STATIC_GRAPH.neighbors(node_id))
            are_empty = map(
                lambda node_id: node_id not in self.buildings,
                under_consideration,
            )
            return all(are_empty)

        # if initial-placement, iterate over non-water/port tiles, for each
        # of these nodes check if its a buildable node.
        if initial_build_phase:
            for (_, tile) in self.map.resource_tiles():
                for (_, node_id) in tile.nodes.items():
                    if is_buildable(node_id):
                        buildable.add(node_id)

        # if not initial-placement, find all connected components. For each
        #   node in this connected subgraph, iterate checking buildability
        subgraphs = self.find_connected_components(color)
        for subgraph in subgraphs:
            for node_id in subgraph:
                # by definition node is "connected", so only need to check buildable
                if is_buildable(node_id):
                    buildable.add(node_id)

        return sorted(list(buildable))

    def buildable_edges(self, color: Color):
        """List of (n1,n2) tuples. Edges are in n1 < n2 order. Result is also ordered."""
        global STATIC_GRAPH
        buildable_subgraph = STATIC_GRAPH.subgraph(range(NUM_NODES))
        expandable = set()

        # non-enemy-nodes in your connected components
        expandable_nodes = set(
            map(
                lambda x: x[0],
                filter(lambda x: x[1][0] == color, self.buildings.items()),
            )
        )
        for node_set in self.connected_components[color]:
            for node in node_set:
                if not self.is_enemy_node(node, color):
                    expandable_nodes.add(node)

        candidate_edges = buildable_subgraph.edges(expandable_nodes)
        for edge in candidate_edges:
            if self.get_edge_color(edge) is None:
                expandable.add(tuple(sorted(edge)))

        return sorted(list(expandable))

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        for resource, node_ids in self.map.get_port_nodes().items():
            for node_id in node_ids:
                if self.get_node_color(node_id) == color:
                    yield resource

    def find_connected_components(self, color: Color):
        """
        Returns:
            nx.Graph[]: connected subgraphs. subgraphs
                might include nodes that color doesnt own (on the way and on ends),
                just to make it is "closed" and easier for buildable_nodes to operate.
        """
        return self.connected_components[color]

    def update_connected_components(self):
        global STATIC_GRAPH
        components = defaultdict(list)
        edge_agenda = set(self.roads.keys())
        while len(edge_agenda) != 0:
            seed = edge_agenda.pop()
            color = self.roads[seed]
            subagenda = [seed]
            visited = set()

            while len(subagenda) != 0:
                edge = subagenda.pop()
                visited.add(edge)
                if edge in edge_agenda:
                    edge_agenda.remove(edge)

                for node in edge:
                    # try to explore in this direction
                    node_color = self.get_node_color(node)
                    if node_color == color or node_color is None:
                        explorable_edges = [
                            tuple(sorted(e)) for e in STATIC_GRAPH.edges(node)
                        ]
                        explorable_edges = [
                            e
                            for e in explorable_edges
                            if e != edge
                            and self.get_edge_color(e) == color
                            and e not in visited
                        ]
                        subagenda.extend(explorable_edges)

            node_set = set()
            for edge in visited:
                node_set.add(edge[0])
                node_set.add(edge[1])

            components[color].append(node_set)

        self.connected_components = components

    # ===== Helper functions
    def get_node_color(self, node_id):
        try:
            return self.buildings[node_id][0]
        except KeyError:
            return None

    def get_edge_color(self, edge):
        try:
            normalized_edge = tuple(sorted(edge))
            return self.roads[normalized_edge]
        except KeyError:
            return None

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color
