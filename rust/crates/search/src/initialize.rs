use catanatron_core::{
    draw_bounded, GameContext, LandTile, Layout, MapKind, NodeId, Port, Position, RandomSource,
    Resource,
};

use crate::{derive_seed, SearchRng, StreamKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumberPlacement {
    OfficialSpiral,
    Random,
}

const SPIRAL_TILE_ORDER: [usize; 19] = [
    16, 17, 18, 15, 11, 6, 2, 1, 0, 3, 7, 12, 13, 14, 10, 5, 4, 8, 9,
];
const SPIRAL_NUMBERS: [u8; 18] = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];
const RANDOM_NUMBERS: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];
const PORT_NODES: [(u8, u8); 9] = [
    (35, 36),
    (38, 39),
    (40, 44),
    (32, 33),
    (45, 47),
    (28, 29),
    (48, 49),
    (25, 26),
    (52, 53),
];

pub fn initialize_base(
    player_count: u8,
    placement: NumberPlacement,
    master_seed: u64,
    game_index: u64,
) -> Result<(GameContext, Position), catanatron_core::IllegalAction> {
    let mut random =
        SearchRng::from_seed(derive_seed(master_seed, game_index, 0, StreamKind::Chance));
    let mut resources = [
        Some(Resource::Wood),
        Some(Resource::Wood),
        Some(Resource::Wood),
        Some(Resource::Wood),
        Some(Resource::Brick),
        Some(Resource::Brick),
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Sheep),
        Some(Resource::Sheep),
        Some(Resource::Sheep),
        Some(Resource::Wheat),
        Some(Resource::Wheat),
        Some(Resource::Wheat),
        Some(Resource::Wheat),
        Some(Resource::Ore),
        Some(Resource::Ore),
        Some(Resource::Ore),
        None,
    ];
    shuffle(&mut resources, &mut random);
    let mut numbers = RANDOM_NUMBERS;
    if placement == NumberPlacement::Random {
        shuffle(&mut numbers, &mut random);
    }
    let mut tiles = [LandTile::DESERT; 19];
    match placement {
        NumberPlacement::Random => {
            let mut number_index = 0;
            for index in 0..19 {
                if let Some(resource) = resources[index] {
                    tiles[index] = LandTile::producing(resource, numbers[number_index]);
                    number_index += 1;
                }
            }
        }
        NumberPlacement::OfficialSpiral => {
            let mut number_index = 0;
            for &index in &SPIRAL_TILE_ORDER {
                if let Some(resource) = resources[index] {
                    tiles[index] = LandTile::producing(resource, SPIRAL_NUMBERS[number_index]);
                    number_index += 1;
                }
            }
        }
    }
    let mut port_resources = [
        Some(Resource::Wood),
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Wheat),
        Some(Resource::Ore),
        None,
        None,
        None,
        None,
    ];
    shuffle(&mut port_resources, &mut random);
    let mut ports = [None; 9];
    for index in 0..9 {
        let (first, second) = PORT_NODES[index];
        ports[index] = Some(Port::new(
            port_resources[index],
            [
                NodeId::new(first).expect("base port node"),
                NodeId::new(second).expect("base port node"),
            ],
        ));
    }
    let context =
        GameContext::new(Layout::new(tiles).expect("valid BASE assignments")).with_ports(ports);
    let mut position = Position::new(player_count)?;
    position.robber = resources
        .iter()
        .position(Option::is_none)
        .expect("one desert") as u8;
    Ok((context, position))
}

const MINI_SPIRAL_TILE_ORDER: [usize; 7] = [1, 6, 5, 4, 3, 2, 0];
const MINI_RANDOM_NUMBERS: [u8; 7] = [3, 4, 5, 6, 8, 9, 10];

pub fn initialize_mini(
    player_count: u8,
    placement: NumberPlacement,
    master_seed: u64,
    game_index: u64,
) -> Result<(GameContext, Position), catanatron_core::IllegalAction> {
    let mut random =
        SearchRng::from_seed(derive_seed(master_seed, game_index, 0, StreamKind::Chance));
    let mut resources = [
        Some(Resource::Wood),
        None,
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Wheat),
        Some(Resource::Wheat),
        Some(Resource::Ore),
    ];
    shuffle(&mut resources, &mut random);
    let mut numbers = MINI_RANDOM_NUMBERS;
    if placement == NumberPlacement::Random {
        shuffle(&mut numbers, &mut random);
    }
    let mut tiles = [LandTile::DESERT; 19];
    let order: &[usize] = match placement {
        NumberPlacement::Random => &[0, 1, 2, 3, 4, 5, 6],
        NumberPlacement::OfficialSpiral => &MINI_SPIRAL_TILE_ORDER,
    };
    let official_numbers = &SPIRAL_NUMBERS[..6];
    let selected_numbers = if placement == NumberPlacement::Random {
        &numbers[..]
    } else {
        official_numbers
    };
    let mut number_index = 0;
    for &index in order {
        if let Some(resource) = resources[index] {
            tiles[index] = LandTile::producing(resource, selected_numbers[number_index]);
            number_index += 1;
        }
    }
    let context = GameContext::new(Layout::new(tiles).expect("valid MINI assignments"));
    let mut position = Position::new_on_map(player_count, MapKind::Mini)?;
    position.robber = resources
        .iter()
        .position(Option::is_none)
        .expect("one desert") as u8;
    Ok((context, position))
}

fn shuffle<T>(values: &mut [T], random: &mut impl RandomSource) {
    for upper in (1..values.len()).rev() {
        let index = draw_bounded(random, (upper + 1) as u64).expect("positive bound") as usize;
        values.swap(upper, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_initialization_is_reproducible_and_has_one_desert() {
        let (first_context, first_position) =
            initialize_base(4, NumberPlacement::OfficialSpiral, 7, 3).unwrap();
        let (second_context, second_position) =
            initialize_base(4, NumberPlacement::OfficialSpiral, 7, 3).unwrap();
        assert_eq!(first_context, second_context);
        assert_eq!(first_position, second_position);
        assert_eq!(
            (0..19)
                .filter(|&raw| first_context
                    .layout
                    .tile(catanatron_core::TileId::new(raw).unwrap())
                    .resource
                    .is_none())
                .count(),
            1
        );
    }

    #[test]
    fn mini_initialization_uses_only_mini_geometry() {
        let (context, position) =
            initialize_mini(2, NumberPlacement::OfficialSpiral, 7, 3).unwrap();
        assert_eq!(position.map, MapKind::Mini);
        assert_eq!(position.map.active_node_mask().count_ones(), 24);
        assert_eq!(position.map.active_edge_mask().count_ones(), 30);
        assert_eq!(position.map.active_tile_mask().count_ones(), 7);
        assert_eq!(
            (0..7)
                .filter(|&raw| context
                    .layout
                    .tile(catanatron_core::TileId::new(raw).unwrap())
                    .resource
                    .is_none())
                .count(),
            1
        );
        assert!((7..19).all(|raw| context
            .layout
            .tile(catanatron_core::TileId::new(raw).unwrap())
            == LandTile::DESERT));
    }
}
