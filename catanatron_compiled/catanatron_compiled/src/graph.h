#ifndef GRAPH_H
#define GRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace catanatron {

typedef uint16_t NodeId;
typedef std::tuple<NodeId, NodeId> Edge;

struct EdgeHash {
  std::size_t operator()(Edge const& edge) const noexcept {
    std::size_t h1 = std::hash<NodeId>{}(std::get<0>(edge));
    std::size_t h2 = std::hash<NodeId>{}(std::get<1>(edge));
    return h1 ^ (h2 << 1);
  }
};

class Graph {
 public:
  Graph(){};

  void add_nodes_from(const std::vector<NodeId>& node_ids);
  void add_edges_from(const std::vector<Edge>& edges);
  Graph subgraph(const std::vector<NodeId>& node_ids);
  std::vector<Edge> edges(const std::unordered_set<NodeId>& node_ids);
  std::vector<Edge>& edges(const NodeId node_id);
  std::vector<Edge> edges();
  std::vector<Edge> edges() const;
  std::vector<NodeId> nodes() const {
    std::vector<NodeId> nodes(nodes_.begin(), nodes_.end());
    return nodes;
  };
  const std::vector<NodeId>& neighbors(const NodeId& node_id);

 private:
  std::unordered_set<NodeId> nodes_;
  std::unordered_map<NodeId, std::vector<NodeId>> neighbors_;
  std::unordered_map<NodeId, std::vector<Edge>> edges_;
};

std::vector<Edge> longest_acyclic_path(
    const std::unordered_map<NodeId, std::tuple<uint8_t, py::object>>&
        buildings,
    const std::unordered_map<Edge, uint8_t, EdgeHash>& roads,
    std::unordered_set<uint16_t>& node_set, uint8_t color, Graph& graph);

}  // namespace catanatron

#endif