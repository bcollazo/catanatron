//! Immutable BASE board geometry exported from the pinned Python implementation.
#[path = "generated/base.rs"]
mod base;
use crate::{EdgeId, NodeId};
pub const BASE_NODE_COUNT: usize = base::BASE_LAND_NODES.len();
pub const BASE_EDGE_COUNT: usize = base::BASE_EDGES.len();
pub fn edge_endpoints(edge: EdgeId) -> (NodeId, NodeId) {
    let (a, b) = base::BASE_EDGES[usize::from(edge.get())];
    (
        NodeId::new(a).expect("generated node"),
        NodeId::new(b).expect("generated node"),
    )
}
pub fn incident(edge: EdgeId, node: NodeId) -> bool {
    let (a, b) = edge_endpoints(edge);
    a == node || b == node
}
pub fn node_neighbors(node: NodeId) -> impl Iterator<Item = NodeId> {
    base::BASE_EDGES.iter().filter_map(move |&(a, b)| {
        if a == node.get() {
            NodeId::new(b).ok()
        } else if b == node.get() {
            NodeId::new(a).ok()
        } else {
            None
        }
    })
}
