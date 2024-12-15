use crate::{enums::Resource, ordered_hashmap::OrderedHashMap};

// Define a new struct to wrap the tuple
pub type Coordinate = (i8, i8, i8);

// Method to add two coordinates
pub fn add_coordinates(a: Coordinate, b: Coordinate) -> Coordinate {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileSlot {
    Land,
    Water,
    NWPort,
    NEPort,
    EPort,
    SEPort,
    SWPort,
    WPort,
}

#[derive(Debug)]
pub struct MapTemplate {
    pub(crate) numbers: Vec<u8>,
    pub(crate) ports: Vec<Option<Resource>>,
    pub(crate) tiles: Vec<Option<Resource>>,

    // Ordered, so that when map is built, we keep the same node-id, edge-id, and tile-id.
    //  that original catanatron uses.
    pub(crate) topology: OrderedHashMap<Coordinate, TileSlot>,
}
