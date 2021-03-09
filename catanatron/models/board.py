from collections import defaultdict

import networkx as nx

from catanatron.models.player import Color
from catanatron.models.map import BaseMap, NUM_NODES, Port, Tile
from catanatron.models.enums import BuildingType

NODE_DISTANCES = None


def get_node_distances():
    global NODE_DISTANCES
    if NODE_DISTANCES is None:
        board = Board()
        NODE_DISTANCES = nx.floyd_warshall(board.nxgraph)

    return NODE_DISTANCES


EDGES = None


def get_edges():
    global EDGES
    if EDGES is None:
        board = Board()
        EDGES = board.nxgraph.subgraph(range(NUM_NODES)).edges()
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

            # Init graph to hold board dynamic state (buildings).
            nxgraph = nx.Graph()
            for tile in self.map.tiles.values():
                nxgraph.add_nodes_from(tile.nodes.values())
                nxgraph.add_edges_from(tile.edges.values())
            self.nxgraph = nxgraph  # buildings are here too.

            # color => nxgraph.edge_subgraph[] (nodes in subgraph includes incidental,
            #   but not-necesarilly owned nodes. might be owned by enemy).
            self.connected_components = defaultdict(list)
            # color => node_id => connected component. contains incident nodes (may not be owned)
            self.color_node_to_subgraphs = defaultdict(dict)

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

        if "building" in self.nxgraph.nodes[node_id]:
            raise ValueError("Invalid Settlement Placement: a building exists there")

        self.nxgraph.nodes[node_id]["building"] = BuildingType.SETTLEMENT
        self.nxgraph.nodes[node_id]["color"] = color

        # Update connected components
        if node_id not in self.color_node_to_subgraphs[color]:
            subgraph = nx.Graph()  # initialize connected component
            subgraph.add_node(node_id)
            self.connected_components[color].append(subgraph)
            self.color_node_to_subgraphs[color][node_id] = subgraph

        # Handle potentially cut of enemies. if node is in a 3-star, cut enemy subgraphs
        for enemy_color, node_to_subgraph in self.color_node_to_subgraphs.items():
            if enemy_color == color:
                continue
            if node_id in node_to_subgraph:
                # take two enemy roads and cut (A, B, C) => (A, B) and (B, C)
                enemy_edges = list(node_to_subgraph[node_id].edges(node_id))
                if len(enemy_edges) != 2:
                    break
                a = next(filter(lambda n: n != node_id, enemy_edges[0]))
                b = next(filter(lambda n: n != node_id, enemy_edges[1]))

                # check if removing node disconnects.
                c_graph = self.color_node_to_subgraphs[enemy_color][node_id]
                c_graph_copy = nx.Graph(c_graph)
                c_graph_copy.remove_node(node_id)  # should be ok to remove our edge too
                connected = list(nx.connected_components(c_graph_copy))
                if len(connected) == 1:
                    break  # cut but did not disconnect entity.

                # definitely a cut, then create two new graphs from previous graph
                a_nodes, b_nodes = connected
                if a not in a_nodes:
                    tmp = a_nodes
                    a_nodes = b_nodes
                    b_nodes = tmp
                a_graph = nx.Graph()
                a_graph.add_edges_from(c_graph.subgraph(a_nodes).edges)
                a_graph.add_edge(a, node_id)
                b_graph = nx.Graph()
                b_graph.add_edges_from(c_graph.subgraph(b_nodes).edges)
                b_graph.add_edge(b, node_id)

                # update self.connected_components and self.color_node_to_subgraphs
                self.connected_components[enemy_color].remove(c_graph)
                self.connected_components[enemy_color].append(a_graph)
                self.connected_components[enemy_color].append(b_graph)
                for n in a_nodes:
                    self.color_node_to_subgraphs[enemy_color][n] = a_graph
                    assert len(list(nx.connected_components(a_graph))) == 1
                for n in b_nodes:
                    self.color_node_to_subgraphs[enemy_color][n] = b_graph
                    assert len(list(nx.connected_components(b_graph))) == 1
                del self.color_node_to_subgraphs[enemy_color][node_id]
                break

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        inverted_edge = (edge[1], edge[0])
        if edge not in buildable and inverted_edge not in buildable:
            raise ValueError("Invalid Road Placement: not connected")

        if "color" in self.nxgraph.edges[edge] is not None:
            raise ValueError("Invalid Road Placement: a road exists there")

        self.nxgraph.edges[edge]["color"] = color

        # Update connected components
        a_graph = (
            None
            if self.is_enemy_node(edge[0], color)
            else self.color_node_to_subgraphs[color].get(edge[0], None)
        )
        b_graph = (
            None
            if self.is_enemy_node(edge[1], color)
            else self.color_node_to_subgraphs[color].get(edge[1], None)
        )
        if a_graph is not None and b_graph is not None and a_graph != b_graph:
            # merge subgraphs into one (i.e. player 'connected' roads)
            self.connected_components[color].remove(a_graph)
            self.connected_components[color].remove(b_graph)
            c_graph = nx.Graph()
            c_graph.add_edges_from(a_graph.edges)
            c_graph.add_edges_from(b_graph.edges)
            c_graph.add_edge(*edge)
            self.connected_components[color].append(c_graph)
            for node_id in c_graph.nodes:
                self.color_node_to_subgraphs[color][node_id] = c_graph
                assert len(list(nx.connected_components(c_graph))) == 1
        elif a_graph is not None and b_graph is not None:  # but a == b
            # player connected same subgraph (no need to add "other" node)
            self.color_node_to_subgraphs[color][edge[0]].add_edge(*edge)
            assert (
                len(
                    list(
                        nx.connected_components(
                            self.color_node_to_subgraphs[color][edge[0]]
                        )
                    )
                )
                == 1
            )
        elif a_graph is not None:  # but b_graph is None
            self.color_node_to_subgraphs[color][edge[0]].add_edge(*edge)
            self.color_node_to_subgraphs[color][edge[1]] = self.color_node_to_subgraphs[
                color
            ][edge[0]]
            assert (
                len(
                    list(
                        nx.connected_components(
                            self.color_node_to_subgraphs[color][edge[0]]
                        )
                    )
                )
                == 1
            )
        else:  # must be a b_graph edge, a_graph is None
            self.color_node_to_subgraphs[color][edge[1]].add_edge(*edge)
            self.color_node_to_subgraphs[color][edge[0]] = self.color_node_to_subgraphs[
                color
            ][edge[1]]
            assert (
                len(
                    list(
                        nx.connected_components(
                            self.color_node_to_subgraphs[color][edge[0]]
                        )
                    )
                )
                == 1
            )

    def build_city(self, color, node_id):
        node = self.nxgraph.nodes[node_id]
        if not (
            node.get("building", None) == BuildingType.SETTLEMENT
            and node.get("color", None) == color
        ):
            raise ValueError("Invalid City Placement: no player settlement there")

        self.nxgraph.nodes[node_id]["building"] = BuildingType.CITY

    def buildable_node_ids(self, color: Color, initial_build_phase=False):
        buildable = set()

        def is_buildable(node_id):
            """true if this and neighboring nodes are empty"""
            under_consideration = [node_id] + list(self.nxgraph.neighbors(node_id))
            are_empty = map(
                lambda nid: "building" not in self.nxgraph.nodes[nid],
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
        buildable_subgraph = self.nxgraph.subgraph(range(NUM_NODES))
        expandable = set()
        for subgraph in self.connected_components[color]:
            non_enemy_nodes = [
                node_id
                for node_id in subgraph
                if not self.is_enemy_node(node_id, color)
            ]
            candidate_edges = buildable_subgraph.edges(non_enemy_nodes)
            for edge in candidate_edges:
                if self.get_edge_color(edge) is None:
                    expandable.add(tuple(sorted(edge)))

        return sorted(list(expandable))

    def get_player_buildings(self, color, building_type=None):
        """Returns list of (node_id, building_type)"""
        buildings = [
            (node_id, self.nxgraph.nodes[node_id]["building"])
            for node_id in self.nxgraph.nodes
            if self.get_node_color(node_id) == color
        ]
        if building_type is not None:
            buildings = filter(lambda node: node[1] == building_type, buildings)

        return list(buildings)

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

    # ===== Helper functions
    def get_node_color(self, node_id):
        return self.nxgraph.nodes[node_id].get("color", None)

    def get_edge_color(self, edge):
        return self.nxgraph.edges[edge].get("color", None)

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color
