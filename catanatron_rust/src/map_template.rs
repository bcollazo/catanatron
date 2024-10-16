use crate::enums::Resource;
use std::collections::HashMap;

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

// We'll represent a Coordinate as a bitfield of 3 numbers.
//  This allows us to create even the 6-8 player base catan map.
pub fn bitfield_coordinate(x: i8, y: i8, z: i8) -> u32 {
    let x_packed = (x as u32) & 0xFF; // Mask to ensure we only take 8 bits
    let y_packed = (y as u32) & 0xFF; // Mask to ensure we only take 8 bits
    let z_packed = (z as u32) & 0xFF; // Mask to ensure we only take 8 bits

    // Shift the second and third values to their respective positions
    (x_packed) | (y_packed << 8) | (z_packed << 16)
}

pub struct MapTemplate {
    numbers: Vec<i8>,
    ports: Vec<Option<Resource>>,
    tiles: Vec<Option<Resource>>,
    topology: HashMap<u32, TileSlot>,
}

#[cfg(test)]
fn unpack_from_bitfield(bitfield: u32) -> (i8, i8, i8) {
    let a = (bitfield & 0xFF) as i8; // Extract the first 8 bits
    let b = ((bitfield >> 8) & 0xFF) as i8; // Extract the second 8 bits
    let c = ((bitfield >> 16) & 0xFF) as i8; // Extract the third 8 bits

    (a, b, c)
}

mod tests {
    use super::*;

    #[test]
    fn test_coordinate_to_key() {
        assert_eq!(bitfield_coordinate(0, 0, 0), 0);
        assert_eq!(bitfield_coordinate(1, 0, 0), 1);
        assert_eq!(bitfield_coordinate(0, 1, 0), 256);
        assert_eq!(bitfield_coordinate(0, 0, 1), 65536);
        assert_eq!(bitfield_coordinate(1, 1, 1), 65793);
    }

    #[test]
    fn test_map_template_creation() {
        let mut topology = HashMap::new();
        // center
        topology.insert(bitfield_coordinate(0, 0, 0), TileSlot::Land);
        // first layer
        topology.insert(bitfield_coordinate(-1, -1, 0), TileSlot::Land);
        topology.insert(bitfield_coordinate(0, -1, 1), TileSlot::Land);
        topology.insert(bitfield_coordinate(-1, 0, 1), TileSlot::Land);
        topology.insert(bitfield_coordinate(-1, 1, 0), TileSlot::Land);
        topology.insert(bitfield_coordinate(0, 1, -1), TileSlot::Land);
        topology.insert(bitfield_coordinate(1, 0, -1), TileSlot::Land);
        // second layer
        topology.insert(bitfield_coordinate(2, -2, 0), TileSlot::Water);
        topology.insert(bitfield_coordinate(1, -2, 1), TileSlot::Water);
        topology.insert(bitfield_coordinate(0, -2, 2), TileSlot::Water);
        topology.insert(bitfield_coordinate(-1, -1, 2), TileSlot::Water);
        topology.insert(bitfield_coordinate(-2, 0, 2), TileSlot::Water);
        topology.insert(bitfield_coordinate(-2, 1, 1), TileSlot::Water);
        topology.insert(bitfield_coordinate(-2, 2, 0), TileSlot::Water);
        topology.insert(bitfield_coordinate(-1, 2, -1), TileSlot::Water);
        topology.insert(bitfield_coordinate(0, 2, -2), TileSlot::Water);
        topology.insert(bitfield_coordinate(1, 1, -2), TileSlot::Water);
        topology.insert(bitfield_coordinate(2, 0, -2), TileSlot::Water);
        topology.insert(bitfield_coordinate(2, -1, -1), TileSlot::Water);

        let map_template = MapTemplate {
            numbers: vec![3, 4, 5, 6, 8, 9, 10],
            ports: vec![],
            tiles: vec![
                Some(Resource::Wood),
                None as Option<Resource>,
                Some(Resource::Brick),
                Some(Resource::Sheep),
                Some(Resource::Wheat),
                Some(Resource::Wheat),
                Some(Resource::Ore),
            ],
            topology,
        };
        assert_eq!(map_template.topology.len(), 19);
    }
}
