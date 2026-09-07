//! Immutable per-game land-tile assignments, separate from copied positions.
use crate::{Resource, BASE_LAND_TILE_COUNT};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LandTile {
    pub resource: Option<Resource>,
    pub number: Option<u8>,
}

impl LandTile {
    pub const DESERT: Self = Self {
        resource: None,
        number: None,
    };

    pub const fn producing(resource: Resource, number: u8) -> Self {
        Self {
            resource: Some(resource),
            number: Some(number),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    tiles: [LandTile; BASE_LAND_TILE_COUNT],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutError {
    InvalidTile(u8),
}

impl Layout {
    pub fn new(tiles: [LandTile; BASE_LAND_TILE_COUNT]) -> Result<Self, LayoutError> {
        for (index, tile) in tiles.iter().enumerate() {
            match (tile.resource, tile.number) {
                (None, None) => {}
                (Some(_), Some(number)) if (2..=12).contains(&number) && number != 7 => {}
                _ => return Err(LayoutError::InvalidTile(index as u8)),
            }
        }
        Ok(Self { tiles })
    }

    pub fn tile(&self, tile: crate::TileId) -> LandTile {
        self.tiles[usize::from(tile.get())]
    }
}

/// Immutable data shared by transitions; copied `Position` values never own it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GameContext {
    pub layout: Layout,
}

impl GameContext {
    pub const fn new(layout: Layout) -> Self {
        Self { layout }
    }
}
