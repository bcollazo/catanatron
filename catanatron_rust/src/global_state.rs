use std::collections::HashMap;

use crate::{
    enums::Resource,
    map_template::{MapTemplate, TileSlot},
    ordered_hashmap::OrderedHashMap,
};

#[derive(Debug)]
pub struct GlobalState {
    pub mini_map_template: MapTemplate,
    pub base_map_template: MapTemplate,
    pub dice_probas: HashMap<u8, f64>,
}

fn build_dice_probas() -> HashMap<u8, f64> {
    let mut probas: HashMap<u8, f64> = HashMap::new();

    // Iterate over two dice rolls
    for i in 1..=6 {
        for j in 1..=6 {
            let sum = i + j;
            let counter = probas.entry(sum).or_insert(0.0);
            *counter += 1.0 / 36.0;
        }
    }

    probas
}

impl Default for GlobalState {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalState {
    pub fn new() -> Self {
        // Mini Map Template
        let mut topology = OrderedHashMap::new();
        // center
        topology.insert((0, 0, 0), TileSlot::Land);
        // first layer
        topology.insert((1, -1, 0), TileSlot::Land);
        topology.insert((0, -1, 1), TileSlot::Land);
        topology.insert((-1, 0, 1), TileSlot::Land);
        topology.insert((-1, 1, 0), TileSlot::Land);
        topology.insert((0, 1, -1), TileSlot::Land);
        topology.insert((1, 0, -1), TileSlot::Land);
        // second layer
        topology.insert((2, -2, 0), TileSlot::Water);
        topology.insert((1, -2, 1), TileSlot::Water);
        topology.insert((0, -2, 2), TileSlot::Water);
        topology.insert((-1, -1, 2), TileSlot::Water);
        topology.insert((-2, 0, 2), TileSlot::Water);
        topology.insert((-2, 1, 1), TileSlot::Water);
        topology.insert((-2, 2, 0), TileSlot::Water);
        topology.insert((-1, 2, -1), TileSlot::Water);
        topology.insert((0, 2, -2), TileSlot::Water);
        topology.insert((1, 1, -2), TileSlot::Water);
        topology.insert((2, 0, -2), TileSlot::Water);
        topology.insert((2, -1, -1), TileSlot::Water);
        let mini_map_template = MapTemplate {
            numbers: vec![3, 4, 5, 6, 8, 9, 10],
            ports: vec![],
            tiles: vec![
                Some(Resource::Wood),
                None,
                Some(Resource::Brick),
                Some(Resource::Sheep),
                Some(Resource::Wheat),
                Some(Resource::Wheat),
                Some(Resource::Ore),
            ],
            topology,
        };
        // Base Map Template
        let mut topology = OrderedHashMap::new();
        // center
        topology.insert((0, 0, 0), TileSlot::Land);
        // first layer
        topology.insert((1, -1, 0), TileSlot::Land);
        topology.insert((0, -1, 1), TileSlot::Land);
        topology.insert((-1, 0, 1), TileSlot::Land);
        topology.insert((-1, 1, 0), TileSlot::Land);
        topology.insert((0, 1, -1), TileSlot::Land);
        topology.insert((1, 0, -1), TileSlot::Land);
        // second layer
        topology.insert((2, -2, 0), TileSlot::Land);
        topology.insert((1, -2, 1), TileSlot::Land);
        topology.insert((0, -2, 2), TileSlot::Land);
        topology.insert((-1, -1, 2), TileSlot::Land);
        topology.insert((-2, 0, 2), TileSlot::Land);
        topology.insert((-2, 1, 1), TileSlot::Land);
        topology.insert((-2, 2, 0), TileSlot::Land);
        topology.insert((-1, 2, -1), TileSlot::Land);
        topology.insert((0, 2, -2), TileSlot::Land);
        topology.insert((1, 1, -2), TileSlot::Land);
        topology.insert((2, 0, -2), TileSlot::Land);
        topology.insert((2, -1, -1), TileSlot::Land);
        // third layer
        topology.insert((3, -3, 0), TileSlot::WPort);
        topology.insert((2, -3, 1), TileSlot::Water);
        topology.insert((1, -3, 2), TileSlot::NWPort);
        topology.insert((0, -3, 3), TileSlot::Water);
        topology.insert((-1, -2, 3), TileSlot::NWPort);
        topology.insert((-2, -1, 3), TileSlot::Water);
        topology.insert((-3, 0, 3), TileSlot::NEPort);
        topology.insert((-3, 1, 2), TileSlot::Water);
        topology.insert((-3, 2, 1), TileSlot::EPort);
        topology.insert((-3, 3, 0), TileSlot::Water);
        topology.insert((-2, 3, -1), TileSlot::EPort);
        topology.insert((-1, 3, -2), TileSlot::Water);
        topology.insert((0, 3, -3), TileSlot::SEPort);
        topology.insert((1, 2, -3), TileSlot::Water);
        topology.insert((2, 1, -3), TileSlot::SWPort);
        topology.insert((3, 0, -3), TileSlot::Water);
        topology.insert((3, -1, -2), TileSlot::SWPort);
        topology.insert((3, -2, -1), TileSlot::Water);

        let base_map_template = MapTemplate {
            numbers: vec![2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
            ports: vec![
                // These are 2:1 ports
                Some(Resource::Wood),
                Some(Resource::Brick),
                Some(Resource::Sheep),
                Some(Resource::Wheat),
                Some(Resource::Ore),
                // These represet 3:1 ports
                None,
                None,
                None,
                None,
            ],
            tiles: vec![
                // Four wood tiles
                Some(Resource::Wood),
                Some(Resource::Wood),
                Some(Resource::Wood),
                Some(Resource::Wood),
                // Three brick tiles
                Some(Resource::Brick),
                Some(Resource::Brick),
                Some(Resource::Brick),
                // Four sheep tiles
                Some(Resource::Sheep),
                Some(Resource::Sheep),
                Some(Resource::Sheep),
                Some(Resource::Sheep),
                // Four wheat tiles
                Some(Resource::Wheat),
                Some(Resource::Wheat),
                Some(Resource::Wheat),
                Some(Resource::Wheat),
                // Three ore tiles
                Some(Resource::Ore),
                Some(Resource::Ore),
                Some(Resource::Ore),
                // One desert
                None,
            ],
            topology,
        };

        let dice_probas = build_dice_probas();

        Self {
            mini_map_template,
            base_map_template,
            dice_probas,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_state() {
        let global_state = GlobalState::new();
        assert_eq!(global_state.mini_map_template.numbers.len(), 7);
        assert_eq!(global_state.base_map_template.numbers.len(), 18);
        assert_eq!(global_state.mini_map_template.topology.len(), 19);
    }
}
