//! Immutable BASE board geometry exported from the pinned Python implementation.
#[path = "generated/base.rs"]
mod base;
use crate::{EdgeId, NodeId};
pub const BASE_NODE_COUNT: usize = base::BASE_LAND_NODES.len();
pub const BASE_EDGE_COUNT: usize = base::BASE_EDGES.len();
pub const BASE_LAND_TILE_COUNT: usize = base::BASE_LAND_TILE_NODES.len();
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

/// Returns the six vertices touching a dense BASE land-tile index.
pub fn land_tile_nodes(tile: crate::TileId) -> [NodeId; 6] {
    let (a, b, c, d, e, f) = base::BASE_LAND_TILE_NODES[usize::from(tile.get())];
    [a, b, c, d, e, f].map(|node| NodeId::new(node).expect("generated node"))
}
