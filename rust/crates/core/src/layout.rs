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
    pub ports: [Option<Port>; 9],
    pub friendly_robber: bool,
}

impl GameContext {
    pub const fn new(layout: Layout) -> Self {
        Self {
            layout,
            ports: [None; 9],
            friendly_robber: false,
        }
    }

    pub const fn with_ports(mut self, ports: [Option<Port>; 9]) -> Self {
        self.ports = ports;
        self
    }

    pub const fn with_friendly_robber(mut self, enabled: bool) -> Self {
        self.friendly_robber = enabled;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Port {
    pub resource: Option<Resource>,
    pub nodes: [crate::NodeId; 2],
}

impl Port {
    pub const fn new(resource: Option<Resource>, nodes: [crate::NodeId; 2]) -> Self {
        Self { resource, nodes }
    }
}

pub fn maritime_rate(
    position: &crate::Position,
    context: &GameContext,
    actor: crate::PlayerId,
    give: Resource,
) -> u8 {
    let mut rate = 4;
    for port in context.ports.iter().flatten() {
        if port.nodes.iter().any(|node| {
            crate::building_belongs_to(position.buildings[usize::from(node.get())], actor)
        }) {
            let port_rate = if port.resource == Some(give) {
                2
            } else if port.resource.is_none() {
                3
            } else {
                4
            };
            rate = rate.min(port_rate);
        }
    }
    rate
}
