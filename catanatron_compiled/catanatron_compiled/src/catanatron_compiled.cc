#include "graph.h"

#include <pybind11/pybind11.h>

namespace catanatron {

// ----------------
// Python interface
// ----------------

PYBIND11_MODULE(catanatron_compiled, m) {
  m.doc() = "Compiled Catanatron features";

  // Add Graph Module
  py::module_ graph_module = m.def_submodule("graph", "Compiled Graph features");
  py::class_<Graph>(graph_module, "Graph")
      .def(py::init<>())
      // Implement pickle serialization
      .def(py::pickle(
          [](const Graph& g) {  // dump
            return py::make_tuple(g.edges(), g.nodes());
          },
          [](py::tuple t) {  // load
            Graph g;
            std::vector<Edge> edges = t[0].cast<std::vector<Edge>>();
            std::vector<NodeId> nodes = t[1].cast<std::vector<NodeId>>();
            g.add_nodes_from(nodes);
            g.add_edges_from(edges);
            return g;
          }))
      .def("add_nodes_from", &Graph::add_nodes_from)
      .def("add_edges_from", &Graph::add_edges_from)
      .def("subgraph", &Graph::subgraph)
      // Use overload_cast to handle overloaded methods.
      .def("edges",
           py::overload_cast<const std::unordered_set<NodeId>&>(&Graph::edges))
      .def("edges", py::overload_cast<const NodeId>(&Graph::edges))
      .def("edges", py::overload_cast<>(&Graph::edges))
      .def("neighbors", &Graph::neighbors);

  graph_module.def("longest_acyclic_path", &longest_acyclic_path);
}

}  // namespace catanatron