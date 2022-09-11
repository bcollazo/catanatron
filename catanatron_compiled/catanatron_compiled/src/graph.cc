#include "graph.h"

namespace catanatron {

void Graph::add_nodes_from(const std::vector<NodeId>& node_ids) {
  for (const auto& node_id : node_ids) {
    nodes_.insert(node_id);
    neighbors_.emplace(node_id, std::vector<NodeId>{});
    edges_.emplace(node_id, std::vector<Edge>{});
  }
}

void Graph::add_edges_from(const std::vector<Edge>& edges) {
  for (const auto& edge : edges) {
    const auto& node_a = std::get<0>(edge);
    const auto& node_b = std::get<1>(edge);

    if (nodes_.find(node_a) == nodes_.end() ||
        nodes_.find(node_b) == nodes_.end()) {
      continue;
    }

    const auto& neighbors_a = neighbors_[node_a];
    const auto& neighbors_b = neighbors_[node_b];

    if (std::find(neighbors_a.begin(), neighbors_a.end(), node_b) ==
        neighbors_a.end()) {
      neighbors_[node_a].push_back(node_b);
    }
    if (std::find(neighbors_b.begin(), neighbors_b.end(), node_a) ==
        neighbors_b.end()) {
      neighbors_[node_b].push_back(node_a);
    }

    const auto& edges_a = edges_[node_a];
    const auto& edges_b = edges_[node_b];

    if (std::find(edges_a.begin(), edges_a.end(), edge) == edges_a.end()) {
      edges_[node_a].push_back(edge);
    }

    if (std::find(edges_b.begin(), edges_b.end(), edge) == edges_b.end()) {
      edges_[node_b].push_back(edge);
    }
  }
}

Graph Graph::subgraph(const std::vector<NodeId>& node_ids) {
  Graph graph;
  graph.add_nodes_from(node_ids);

  std::unordered_set<Edge, EdgeHash> edges;
  for (const auto& node_id : node_ids) {
    for (const auto& node_edge : edges_[node_id]) {
      edges.insert(node_edge);
    }
  }

  std::vector<Edge> unique_edges(edges.begin(), edges.end());
  graph.add_edges_from(unique_edges);
  return graph;
}

std::vector<Edge> Graph::edges(const std::unordered_set<NodeId>& node_ids) {
  std::unordered_set<Edge, EdgeHash> edges;
  for (const auto& node_id : node_ids) {
    for (const auto& node_edge : edges_[node_id]) {
      edges.insert(node_edge);
    }
  }
  std::vector<Edge> unique_edges(edges.begin(), edges.end());
  return unique_edges;
}

std::vector<Edge>& Graph::edges(NodeId node_id) { return edges_[node_id]; }

std::vector<Edge> Graph::edges() {
  std::unordered_set<Edge, EdgeHash> edges;
  for (const auto& node_edges : edges_) {
    for (const auto& node_edge : node_edges.second) {
      edges.insert(node_edge);
    }
  }

  std::vector<Edge> unique_edges(edges.begin(), edges.end());
  return unique_edges;
}

std::vector<Edge> Graph::edges() const {
  std::unordered_set<Edge, EdgeHash> edges;
  for (const auto& node_edges : edges_) {
    for (const auto& node_edge : node_edges.second) {
      edges.insert(node_edge);
    }
  }

  std::vector<Edge> unique_edges(edges.begin(), edges.end());
  return unique_edges;
}

const std::vector<NodeId>& Graph::neighbors(const NodeId& node_id) {
  return neighbors_[node_id];
}

std::vector<Edge> longest_acyclic_path(
    const std::unordered_map<NodeId, std::tuple<uint8_t, py::object>>&
        buildings,
    const std::unordered_map<Edge, uint8_t, EdgeHash>& roads,
    std::unordered_set<uint16_t>& node_set, uint8_t color, Graph& graph) {
  uint8_t max_length = 0;
  std::vector<Edge> max_path = {};

  for (const auto& start_node : node_set) {
    // Perform a DFS search until we reach a leaf node and then add to paths.
    std::vector<std::tuple<NodeId, std::vector<Edge>>> agenda = {
        std::make_tuple(start_node, std::vector<Edge>{})};

    while (agenda.size() > 0) {
      auto item = agenda.back();
      agenda.pop_back();

      auto node = std::get<0>(item);
      std::vector<Edge>& path_thus_far = std::get<1>(item);

      bool able_to_navigate = false;

      for (const NodeId& neighbor : graph.neighbors(node)) {
        const auto& neighbor_edge = std::make_tuple(node, neighbor);
        if (roads.find(neighbor_edge) == roads.end()) continue;
        uint8_t edge_color = roads.at(neighbor_edge);

        if (edge_color != color) {
          continue;
        }

        if (buildings.find(neighbor) != buildings.end()) {
          uint8_t neighbor_color = std::get<0>(buildings.at(neighbor));

          if (neighbor_color != color) {
            // enemy owned. Can't use this node to navigate
            continue;
          }
        }
        auto edge = node < neighbor ? std::make_tuple(node, neighbor)
                                    : std::make_tuple(neighbor, node);
        if (std::find(path_thus_far.begin(), path_thus_far.end(), edge) ==
            path_thus_far.end()) {
          able_to_navigate = true;
          std::vector<Edge> new_path_thus_far = path_thus_far;
          new_path_thus_far.push_back(edge);
          agenda.insert(agenda.begin(),
                        std::make_tuple(neighbor, new_path_thus_far));
        }
      }

      if (!able_to_navigate && path_thus_far.size() > max_length) {
        max_length = path_thus_far.size();
        max_path = path_thus_far;
      }
    }
  }
  return max_path;
}

}  // namespace catanatron