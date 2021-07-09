import pickle
from collections import defaultdict
from typing import Set, Tuple

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
        EDGES = list(STATIC_GRAPH.subgraph(range(NUM_NODES)).edges())
    return EDGES


EdgeId = Tuple[int, int]
NodeId = int
Coordinate = Tuple[int, int, int]


class Board:
    """Encapsulates all state information regarding the board.

    Attributes:
        buildings (Dict[NodeId, Tuple[Color, BuildingType]]): Mapping from
            node id to building (if there is a building there).
        roads (Dict[EdgeId, Color]): Mapping from edge
            to Color (if there is a road there). Contains inverted
            edges as well for ease of querying.
        connected_components (Dict[Color, List[Set[NodeId]]]): Cache
            datastructure to speed up mantaining longest road computation.
            To be queried by Color. Value is a list of node sets.
        board_buildable_ids (Set[NodeId]): Cache of buildable node ids in board.
        road_color (Color): Color of player with longest road.
        road_length (int): Number of roads of longest road
        robber_coordinate (Coordinate): Coordinate where robber is.
    """

    def __init__(self, catan_map=None, initialize=True):
        if initialize:
            self.map = catan_map or BaseMap()  # Static State (no need to copy)

            self.buildings = dict()  # node_id => (color, building_type)
            self.roads = dict()  # (node_id, node_id) => color

            # color => int{}[] (list of node_id sets) one per component
            #   nodes in sets are incidental (might not be owned by player)
            self.connected_components = defaultdict(list)
            self.board_buildable_ids = set(range(NUM_NODES))
            self.road_lengths = defaultdict(int)
            self.road_color = None
            self.road_length = 0

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

        previous_road_color = self.road_color
        if initial_build_phase:
            self.connected_components[color].append(set([node_id]))
        else:
            # Maybe cut connected components.
            edges_by_color = defaultdict(list)
            for edge in STATIC_GRAPH.edges(node_id):
                edges_by_color[self.roads.get(edge, None)].append(edge)

            for edge_color, edges in edges_by_color.items():
                if edge_color == color or edge_color is None:
                    continue  # ignore
                if len(edges) == 2:  # rip, edge_color has been plowed
                    # consider cut was at b=node_id for edges (a, b) and (b, c)
                    a = [n for n in edges[0] if n != node_id].pop()
                    c = [n for n in edges[1] if n != node_id].pop()

                    # do bfs from a adding all encountered nodes
                    a_nodeset = self.bfs_walk(a, edge_color)
                    c_nodeset = self.bfs_walk(c, edge_color)

                    # split this components on here.
                    b_index = self._get_connected_component_index(node_id, edge_color)
                    del self.connected_components[edge_color][b_index]
                    self.connected_components[edge_color].append(a_nodeset)
                    self.connected_components[edge_color].append(c_nodeset)

                    # Update longest road by plowed player. Compare again with all
                    self.road_lengths[edge_color] = max(
                        *[
                            len(longest_acyclic_path(self, component, edge_color))
                            for component in self.connected_components[edge_color]
                        ]
                    )
                    self.road_color, self.road_length = max(
                        self.road_lengths.items(), key=lambda e: e[1]
                    )

        self.board_buildable_ids.discard(node_id)
        for n in STATIC_GRAPH.neighbors(node_id):
            self.board_buildable_ids.discard(n)

        return previous_road_color, self.road_color, self.road_lengths

    def bfs_walk(self, node_id, color):
        """Generates set of nodes that are "connected" to given node.

        Args:
            node_id (int): Where to start search/walk.
            color (Color): Player color asking

        Returns:
            Set[int]: Nodes that are "connected" to this one
                by roads of the color player.
        """
        agenda = [node_id]  # assuming node_id is owned.
        visited = set()

        while len(agenda) != 0:
            n = agenda.pop()
            visited.add(n)

            if self.is_enemy_node(n, color):
                continue  # end of the road

            neighbors = [v for v in STATIC_GRAPH.neighbors(n) if v not in visited]
            expandable = [v for v in neighbors if self.roads.get((n, v), None) == color]
            agenda.extend(expandable)

        return visited

    def _get_connected_component_index(self, node_id, color):
        for i, component in enumerate(self.connected_components[color]):
            if node_id in component:
                return i

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        inverted_edge = (edge[1], edge[0])
        if edge not in buildable and inverted_edge not in buildable:
            raise ValueError("Invalid Road Placement")

        self.roads[edge] = color
        self.roads[inverted_edge] = color

        # Update self.connected_components accordingly. Maybe merge.
        a, b = edge
        a_index = self._get_connected_component_index(a, color)
        b_index = self._get_connected_component_index(b, color)
        if a_index is None and b_index is not None and not self.is_enemy_node(a, color):
            self.connected_components[color][b_index].add(a)
            component = self.connected_components[color][b_index]
        elif (
            a_index is not None and b_index is None and not self.is_enemy_node(b, color)
        ):
            self.connected_components[color][a_index].add(b)
            component = self.connected_components[color][a_index]
        elif a_index is not None and b_index is not None and a_index != b_index:
            # merge
            merged_component = self.connected_components[color][a_index].union(
                self.connected_components[color][b_index]
            )
            for index in sorted([a_index, b_index], reverse=True):
                del self.connected_components[color][index]
            self.connected_components[color].append(merged_component)
            component = merged_component
        else:  # both nodes in same component; got nothing to do (already added)
            component = self.connected_components[color][
                a_index if a_index is not None else b_index
            ]

        # find longest path on component under question
        previous_road_color = self.road_color
        candidate_length = len(longest_acyclic_path(self, component, color))
        self.road_lengths[color] = max(self.road_lengths[color], candidate_length)
        if candidate_length >= 5 and candidate_length > self.road_length:
            self.road_color = color
            self.road_length = candidate_length
        return previous_road_color, self.road_color, self.road_lengths

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
        if initial_build_phase:
            return sorted(list(self.board_buildable_ids))

        subgraphs = self.find_connected_components(color)
        nodes = set().union(*subgraphs)
        return sorted(list(nodes.intersection(self.board_buildable_ids)))

    def buildable_edges(self, color: Color):
        """List of (n1,n2) tuples. Edges are in n1 < n2 order."""
        global STATIC_GRAPH
        buildable_subgraph = STATIC_GRAPH.subgraph(range(NUM_NODES))
        expandable = set()

        # non-enemy-nodes in your connected components
        expandable_nodes = set()
        for node_set in self.connected_components[color]:
            for node in node_set:
                if not self.is_enemy_node(node, color):
                    expandable_nodes.add(node)

        candidate_edges = buildable_subgraph.edges(expandable_nodes)
        for edge in candidate_edges:
            if self.get_edge_color(edge) is None:
                expandable.add(tuple(sorted(edge)))

        return list(expandable)

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        for resource, node_ids in self.map.port_nodes.items():
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

    def continuous_roads_by_player(self, color: Color):
        paths = []
        components = self.find_connected_components(color)
        for component in components:
            paths.append(longest_acyclic_path(self, component, color))
        return paths

    def copy(self):
        board = Board(self.map, initialize=False)
        board.map = self.map  # reuse since its immutable
        board.buildings = self.buildings.copy()
        board.roads = self.roads.copy()
        board.connected_components = pickle.loads(
            pickle.dumps(self.connected_components)
        )
        board.board_buildable_ids = self.board_buildable_ids.copy()
        board.road_lengths = self.road_lengths.copy()
        board.road_color = self.road_color
        board.road_length = self.road_length

        board.robber_coordinate = self.robber_coordinate
        return board

    # ===== Helper functions
    def get_node_color(self, node_id):
        try:
            return self.buildings[node_id][0]
        except KeyError:
            return None

    def get_edge_color(self, edge):
        try:
            return self.roads[edge]
        except KeyError:
            return None

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color

    def is_enemy_road(self, edge, color):
        edge_color = self.get_edge_color(edge)
        return edge_color is not None and edge_color != color


def longest_acyclic_path(board: Board, node_set: Set[int], color: Color):
    global STATIC_GRAPH

    paths = []
    for start_node in node_set:
        # do DFS when reach leaf node, stop and add to paths
        paths_from_this_node = []
        agenda = [(start_node, [])]
        while len(agenda) > 0:
            node, path_thus_far = agenda.pop()

            able_to_navigate = False
            for neighbor_node in STATIC_GRAPH.neighbors(node):
                edge_color = board.get_edge_color((node, neighbor_node))
                if edge_color != color:
                    continue

                neighbor_color = board.get_node_color(neighbor_node)
                if neighbor_color is not None and neighbor_color != color:
                    continue  # enemy-owned, cant use this to navigate.
                edge = tuple(sorted((node, neighbor_node)))
                if edge not in path_thus_far:
                    agenda.insert(0, (neighbor_node, path_thus_far + [edge]))
                    able_to_navigate = True

            if not able_to_navigate:  # then it is leaf node
                paths_from_this_node.append(path_thus_far)

        paths.extend(paths_from_this_node)

    return max(paths, key=len)
