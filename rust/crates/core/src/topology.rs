//! Immutable BASE board geometry exported from the pinned Python implementation.
#[path = "generated/base.rs"]
mod base;
#[path = "generated/mini.rs"]
mod mini;
use crate::{EdgeId, NodeId};
pub const BASE_NODE_COUNT: usize = base::BASE_LAND_NODES.len();
pub const BASE_EDGE_COUNT: usize = base::BASE_EDGES.len();
pub const BASE_LAND_TILE_COUNT: usize = base::BASE_LAND_TILE_NODES.len();
pub const MINI_NODE_COUNT: usize = mini::MINI_LAND_NODES.len();
pub const MINI_EDGE_COUNT: usize = mini::MINI_EDGES.len();
pub const MINI_LAND_TILE_COUNT: usize = mini::MINI_LAND_TILE_NODES.len();

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MapKind {
    #[default]
    Base,
    Mini,
}

impl MapKind {
    pub const fn node_count(self) -> usize {
        match self {
            Self::Base => BASE_NODE_COUNT,
            Self::Mini => MINI_NODE_COUNT,
        }
    }

    pub const fn edge_count(self) -> usize {
        match self {
            Self::Base => BASE_EDGE_COUNT,
            Self::Mini => MINI_EDGE_COUNT,
        }
    }

    pub const fn land_tile_count(self) -> usize {
        match self {
            Self::Base => BASE_LAND_TILE_COUNT,
            Self::Mini => MINI_LAND_TILE_COUNT,
        }
    }

    pub const fn active_node_mask(self) -> u64 {
        match self {
            Self::Base => base::BASE_ACTIVE_NODE_MASK,
            Self::Mini => mini::MINI_ACTIVE_NODE_MASK,
        }
    }

    pub const fn active_edge_mask(self) -> u128 {
        match self {
            Self::Base => base::BASE_ACTIVE_EDGE_MASK,
            Self::Mini => mini::MINI_ACTIVE_EDGE_MASK,
        }
    }

    pub const fn active_tile_mask(self) -> u32 {
        match self {
            Self::Base => base::BASE_ACTIVE_TILE_MASK,
            Self::Mini => mini::MINI_ACTIVE_TILE_MASK,
        }
    }
}

fn edges(map: MapKind) -> &'static [(u8, u8)] {
    match map {
        MapKind::Base => base::BASE_EDGES,
        MapKind::Mini => mini::MINI_EDGES,
    }
}

pub fn edge_endpoints_on(map: MapKind, edge: EdgeId) -> Option<(NodeId, NodeId)> {
    let &(a, b) = edges(map).get(usize::from(edge.get()))?;
    Some((
        NodeId::new(a).expect("generated node"),
        NodeId::new(b).expect("generated node"),
    ))
}

pub fn incident_on(map: MapKind, edge: EdgeId, node: NodeId) -> bool {
    edge_endpoints_on(map, edge).is_some_and(|(a, b)| a == node || b == node)
}

pub fn node_neighbors_on(map: MapKind, node: NodeId) -> impl Iterator<Item = NodeId> {
    edges(map).iter().filter_map(move |&(a, b)| {
        if a == node.get() {
            NodeId::new(b).ok()
        } else if b == node.get() {
            NodeId::new(a).ok()
        } else {
            None
        }
    })
}

pub fn land_tile_nodes_on(map: MapKind, tile: crate::TileId) -> Option<[NodeId; 6]> {
    let row = match map {
        MapKind::Base => base::BASE_LAND_TILE_NODES.get(usize::from(tile.get())),
        MapKind::Mini => mini::MINI_LAND_TILE_NODES.get(usize::from(tile.get())),
    }?;
    let &(a, b, c, d, e, f) = row;
    Some([a, b, c, d, e, f].map(|node| NodeId::new(node).expect("generated node")))
}
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
