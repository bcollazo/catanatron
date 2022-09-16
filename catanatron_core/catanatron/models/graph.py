from catanatron_compiled import graph
import networkx as nx


class HybridGraph:
    def __init__(self):
        self.NX_GRAPH = nx.Graph()
        self.CPP_GRAPH = graph.Graph()

    def floyd_warshall(self):
        return nx.floyd_warshall(self.NX_GRAPH)

    def shortest_path(self, a, b):
        return nx.shortest_path(self.NX_GRAPH, a, b)

    def edge_subgraph(self, edges):
        return self.NX_GRAPH.edge_subgraph(edges)

    def add_nodes_from(self, nodes):
        self.NX_GRAPH.add_nodes_from(nodes)

        # Ensure nodes are passed in as a list since the C++ bindings don't
        # automatically translate Python's iterable types to vectors.
        self.CPP_GRAPH.add_nodes_from(list(nodes))

    def add_edges_from(self, edges):
        self.NX_GRAPH.add_edges_from(edges)

        # Ensure edges are pass in as a list since the C++ bindings don't
        # automatically translate Python's iterable types to vectors.
        self.CPP_GRAPH.add_edges_from(list(edges))

    def subgraph(self, nodes):
        return self.CPP_GRAPH.subgraph(list(nodes))

    def edges(self, node_id):
        return self.CPP_GRAPH.edges(node_id)

    def neighbors(self, node_id):
        return self.CPP_GRAPH.neighbors(node_id)
