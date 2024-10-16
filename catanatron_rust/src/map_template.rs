use crate::enums::Resource;
use std::collections::HashMap;

#[derive(Debug)]
pub enum TileSlot {
    Land,
    Water,
    NWPort,
    NPort,
    NEPort,
    EPort,
    SEPort,
    SPort,
    SWPort,
    WPort,
}

type Coordinate = (i8, i8, i8);

#[derive(Debug)]
pub struct MapTemplate {
    pub(crate) numbers: Vec<i8>,
    pub(crate) ports: Vec<Option<Resource>>,
    pub(crate) tiles: Vec<Option<Resource>>,
    pub(crate) topology: HashMap<Coordinate, TileSlot>,
}
