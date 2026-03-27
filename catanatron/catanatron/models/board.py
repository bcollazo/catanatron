import pickle
import copy
from collections import defaultdict
from typing import Any, Set, Dict, Tuple, List
import functools

import networkx as nx  # type: ignore

from catanatron.models.player import Color
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    NodeId,
)
from catanatron.models.enums import FastBuildingType, SETTLEMENT, CITY


# Used to find relationships between nodes and edges
base_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
mini_map = CatanMap.from_template(MINI_MAP_TEMPLATE)
STATIC_GRAPH = nx.Graph()
for tile in base_map.tiles.values():
    STATIC_GRAPH.add_nodes_from(tile.nodes.values())
    STATIC_GRAPH.add_edges_from(tile.edges.values())


@functools.lru_cache(1)
def get_node_distances():
    return nx.floyd_warshall(STATIC_GRAPH)


@functools.lru_cache(3)  # None, range(54), range(24)
def get_edges(land_nodes=None):
    return list(STATIC_GRAPH.subgraph(land_nodes or range(NUM_NODES)).edges())


class Board:
    """Encapsulates all state information regarding the board.

    Attributes:
        buildings (Dict[NodeId, Tuple[Color, FastBuildingType]]): Mapping from
            node id to building (if there is a building there).
        roads (Dict[EdgeId, Color]): Mapping from edge
            to Color (if there is a road there). Contains inverted
            edges as well for ease of querying.
        connected_components (Dict[Color, List[Set[NodeId]]]): Cache
            datastructure to speed up maintaining longest road computation.
            To be queried by Color. Value is a list of node sets.
        board_buildable_ids (Set[NodeId]): Cache of buildable node ids in board.
        road_color (Color): Color of player with longest road.
        road_length (int): Number of roads of longest road
        robber_coordinate (Coordinate): Coordinate where robber is.
    """

    def __init__(self, catan_map=None, initialize=True):
        self.buildable_subgraph: Any = None
        self.buildable_edges_cache = {}
        self.player_port_resources_cache = {}
        if initialize:
            self.map: CatanMap = catan_map or CatanMap.from_template(
                BASE_MAP_TEMPLATE
            )  # Static State (no need to copy)

            self.buildings: Dict[NodeId, Tuple[Color, FastBuildingType]] = dict()
            self.roads = dict()  # (node_id, node_id) => color

            # color => int{}[] (list of node_id sets) one per component
            #   nodes in sets are incidental (might not be owned by player)
            self.connected_components: Any = defaultdict(list)
            self.board_buildable_ids = set(self.map.land_nodes)
            self.road_lengths = defaultdict(int)
            self.road_color = None
            self.road_length = 0

            # assumes there is at least one desert:
            self.robber_coordinate = filter(
                lambda coordinate: self.map.land_tiles[coordinate].resource is None,
                self.map.land_tiles.keys(),
            ).__next__()

            # Cache buildable subgraph
            self.buildable_subgraph = STATIC_GRAPH.subgraph(self.map.land_nodes)

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

        self.buildings[node_id] = (color, SETTLEMENT)

        previous_road_color = self.road_color
        if initial_build_phase:
            self.connected_components[color].append({node_id})
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

                    # do dfs from a adding all encountered nodes
                    a_nodeset = self.dfs_walk(a, edge_color)
                    c_nodeset = self.dfs_walk(c, edge_color)

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

        self.buildable_edges_cache = {}  # Reset buildable_edges
        self.player_port_resources_cache = {}  # Reset port resources
        return previous_road_color, self.road_color, self.road_lengths

    def dfs_walk(self, node_id, color):
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

        # Find connected components corresponding to edge nodes (buildings).
        a, b = edge
        a_index = self._get_connected_component_index(a, color)
        b_index = self._get_connected_component_index(b, color)

        # Extend or merge components
        if a_index is None and not self.is_enemy_node(a, color):
            component = self.connected_components[color][b_index]
            component.add(a)
        elif b_index is None and not self.is_enemy_node(b, color):
            component = self.connected_components[color][a_index]
            component.add(b)
        elif a_index is not None and b_index is not None and a_index != b_index:
            # Merge both components into one and delete the other.
            component = set.union(
                self.connected_components[color][a_index],
                self.connected_components[color][b_index],
            )
            self.connected_components[color][a_index] = component
            del self.connected_components[color][b_index]
        else:
            # In this case, a_index == b_index, which means that the edge
            # is already part of one component. No actions needed.
            chosen_index = a_index if a_index is not None else b_index
            component = self.connected_components[color][chosen_index]

        # find longest path on component under question
        previous_road_color = self.road_color

        # Leaf extension fast path: if one endpoint of the new road had no
        # prior roads of this color, only DFS from the new tip.
        if a_index is None or b_index is None:
            tip = a if b_index is not None else b
            candidate_length = len(longest_acyclic_path(self, component, color, only_from=tip))
        else:
            candidate_length = len(longest_acyclic_path(self, component, color))
        self.road_lengths[color] = max(self.road_lengths[color], candidate_length)
        if candidate_length >= 5 and candidate_length > self.road_length:
            self.road_color = color
            self.road_length = candidate_length

        self.buildable_edges_cache = {}  # Reset buildable_edges
        return previous_road_color, self.road_color, self.road_lengths

    def build_city(self, color, node_id):
        building = self.buildings.get(node_id, None)
        if building is None or building[0] != color or building[1] != SETTLEMENT:
            raise ValueError("Invalid City Placement: no player settlement there")

        self.buildings[node_id] = (color, CITY)

    def buildable_node_ids(self, color: Color, initial_build_phase=False):
        if initial_build_phase:
            return sorted(list(self.board_buildable_ids))

        subgraphs = self.find_connected_components(color)
        nodes = set().union(*subgraphs)
        return sorted(list(nodes.intersection(self.board_buildable_ids)))

    def buildable_edges(self, color: Color):
        """List of (n1,n2) tuples. Edges are in n1 < n2 order."""
        if color in self.buildable_edges_cache:
            return self.buildable_edges_cache[color]

        expandable = set()

        # All nodes for this color.
        # TODO(tonypr): Explore caching for 'expandable_nodes'?
        # The 'expandable_nodes' set should only increase in size monotonically I think.
        # We can take advantage of that.
        expandable_nodes = set()
        expandable_nodes = expandable_nodes.union(*self.connected_components[color])

        candidate_edges = self.buildable_subgraph.edges(expandable_nodes)
        for edge in candidate_edges:
            if self.get_edge_color(edge) is None:
                expandable.add(tuple(sorted(edge)))

        self.buildable_edges_cache[color] = list(expandable)
        return self.buildable_edges_cache[color]

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        if color in self.player_port_resources_cache:
            return self.player_port_resources_cache[color]

        resources = set()
        for resource, node_ids in self.map.port_nodes.items():
            if any(self.is_friendly_node(node_id, color) for node_id in node_ids):
                resources.add(resource)

        self.player_port_resources_cache[color] = resources
        return resources

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
        board.buildable_subgraph = self.buildable_subgraph
        board.buildable_edges_cache = copy.deepcopy(self.buildable_edges_cache)
        board.player_port_resources_cache = copy.deepcopy(
            self.player_port_resources_cache
        )
        return board

    # ===== Helper functions
    def get_node_color(self, node_id):
        # using try-except instead of .get for performance
        try:
            return self.buildings[node_id][0]
        except KeyError:
            return None

    def get_edge_color(self, edge):
        # using try-except instead of .get for performance
        try:
            return self.roads[edge]
        except KeyError:
            return None

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color

    def is_enemy_road(self, edge, color):
        edge_color = self.get_edge_color(edge)
        return edge_color is not None and self.get_edge_color(edge) != color

    def is_friendly_node(self, node_id, color):
        return self.get_node_color(node_id) == color

    def is_friendly_road(self, edge, color):
        return self.get_edge_color(edge) == color


def _road_degree(board: Board, node: int, color: Color) -> int:
    """Count friendly road edges incident to node (that aren't blocked by enemy)."""
    deg = 0
    for neighbor in STATIC_GRAPH.neighbors(node):
        edge = (node, neighbor) if node < neighbor else (neighbor, node)
        if board.is_friendly_road(edge, color):
            deg += 1
    return deg


def longest_acyclic_path(board: Board, node_set: Set[int], color: Color, only_from=None):
    """Find the longest acyclic path of friendly roads in node_set.

    Args:
        only_from: If provided, only start DFS from this node (leaf extension
            fast path). The full node_set is still used for edge discovery.
    """
    # Collect friendly edges in this component
    friendly_edges = set()
    for node in node_set:
        for neighbor in STATIC_GRAPH.neighbors(node):
            if neighbor not in node_set:
                continue
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if board.is_friendly_road(edge, color):
                friendly_edges.add(edge)

    if not friendly_edges:
        return []

    if only_from is not None:
        start_nodes = [only_from]
    else:
        # Start DFS only from leaves and branch points (degree != 2).
        # Optimal path endpoints must be at such vertices.
        # Enemy nodes with friendly roads act as dead ends (effective leaves)
        # and must be included as potential start nodes.
        start_nodes = []
        for node in node_set:
            deg = _road_degree(board, node, color)
            if deg == 0:
                continue
            if board.is_enemy_node(node, color) or deg != 2:
                start_nodes.append(node)

        # Pure-cycle fallback: if all nodes have degree 2, pick any node with roads.
        if not start_nodes:
            for node in node_set:
                if not board.is_enemy_node(node, color) and _road_degree(board, node, color) > 0:
                    start_nodes.append(node)
                    break

    if not start_nodes:
        return []

    best_length = 0
    best_path: list = []

    def _sorted_edge(a, b):
        return (a, b) if a < b else (b, a)

    def _neighbor_iter(node):
        for nb in STATIC_GRAPH.neighbors(node):
            e = _sorted_edge(node, nb)
            if e in friendly_edges and not board.is_enemy_node(nb, color):
                yield nb, e

    for start_node in start_nodes:
        # Iterative DFS with visited-set mutation
        visited: Set[Tuple[int, int]] = set()
        path: List[Tuple[int, int]] = []
        stack = [(start_node, _neighbor_iter(start_node))]

        while stack:
            node, neighbors = stack[-1]

            expanded = False
            for neighbor, edge in neighbors:
                if edge in visited:
                    continue
                visited.add(edge)
                path.append(edge)
                expanded = True
                stack.append((neighbor, _neighbor_iter(neighbor)))
                break

            if not expanded:
                # Leaf: check if this path is the longest so far
                if len(path) > best_length:
                    best_length = len(path)
                    best_path = list(path)
                stack.pop()
                if path:
                    removed = path.pop()
                    visited.discard(removed)

    return best_path
