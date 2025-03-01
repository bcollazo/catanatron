use crate::decks::starting_dev_listdeck;
use rand::seq::SliceRandom;

use crate::enums::COLORS;

pub type StateVector = Vec<u8>;

// Board dimensions (BASE_MAP)
pub const NUM_NODES: usize = 54;
pub const NUM_EDGES: usize = 72;
pub const NUM_TILES: usize = 19;
pub const NUM_PORTS: usize = 9;

// Bank indices
pub const BANK_RESOURCE_SLICE: std::ops::Range<usize> = 0..5;
pub const DEV_BANK_START_INDEX: usize = 5;
pub const DEV_BANK_END_INDEX: usize = 30;
pub const DEV_BANK_PTR_INDEX: usize = 30;

// Game control indices
pub const CURRENT_TICK_SEAT_INDEX: usize = 31;
pub const CURRENT_TURN_SEAT_INDEX: usize = 32;
pub const IS_INITIAL_BUILD_PHASE_INDEX: usize = 33;
pub const HAS_PLAYED_DEV_CARD: usize = 34;
pub const HAS_ROLLED_INDEX: usize = 35;
pub const IS_DISCARDING_INDEX: usize = 36;
pub const IS_MOVING_ROBBER_INDEX: usize = 37;
pub const IS_BUILDING_ROAD_INDEX: usize = 38;
pub const FREE_ROADS_AVAILABLE_INDEX: usize = 39;

// Extra state indices
pub const LONGEST_ROAD_PLAYER_INDEX: usize = 40;
pub const LARGEST_ARMY_PLAYER_INDEX: usize = 41;
pub const ROBBER_TILE_INDEX: usize = 42;

// Player state indices and sizes
pub const PLAYER_STATE_START_INDEX: usize = 268;
pub const PLAYER_STATE_SIZE: usize = 15; // Size of each player's state block
pub const PLAYER_VP_OFFSET: usize = 0;
pub const PLAYER_RESOURCES_OFFSET: usize = 1;
pub const PLAYER_RESOURCES_SIZE: usize = 5;
pub const PLAYER_DEVCARDS_OFFSET: usize = 6;
pub const PLAYER_DEVCARDS_SIZE: usize = 5;
pub const PLAYER_PLAYED_DEVCARDS_OFFSET: usize = 11;
pub const PLAYER_PLAYED_DEVCARDS_SIZE: usize = 4;

// Resource constants
pub const MAX_RESOURCE_COUNT: u8 = 19;
pub const MAX_DEV_CARDS: usize = 25;
pub const MAX_VICTORY_POINTS: u8 = 12;
pub const NUM_RESOURCES: usize = 5;
pub const FREE_ROADS_MAX: u8 = 2;

/// This is in theory not needed since we use a vector and we can
/// .push() to it. But since we made it, leaving in here in case
/// we want to switch to an array implementation and it serves
/// as documentation of the state vector.
pub fn get_state_array_size(num_players: usize) -> usize {
    // TODO: Is configuration part of state?
    // TODO: Hardcoded for BASE_MAP
    let n = num_players;

    log::debug!("get_state_array_size: num_players={}, n={}", num_players, n);

    let mut size: usize = 0;
    // Trying to have as most fixed-size vector first as possible
    //  so that we can understand/debug all configurations similarly.
    // Bank
    size += NUM_RESOURCES; // Bank Resources
    size += MAX_DEV_CARDS; // Bank Development Cards

    // Game Controls
    size += 1; // Current_Player_Index (Player Index < n)
    size += 1; // Current_Turn_Index (Player Index < n)
    size += 1; // Is_Initial_Build_Phase (Boolean)
    size += 1; // Has_Played_Development_Card (Boolean)
    size += 1; // Has_Rolled (Boolean)
    size += 1; // Is_Discarding (Boolean)
    size += 1; // Is_Moving_Robber (Boolean)
    size += 1; // Is_Building_Road (Boolean)
    size += 1; // Free_Roads_Available (Number <= 2)

    // Extra (these are needed to make game Markovian (i.e. memoryless))
    // Note: (Largest_Army_Size and Longest_Road_Size are captured by player (<devcard>_Played) and board state)
    size += 1; // Longest_Road_Player_Index (Player Index < n)
    size += 1; // Largest_Army_Player_Index (Player Index < n)
    size += 1; // Robber_Tile (Tile Index < num_tiles)

    // Board (dynamically sized based on map template; 228 for BASE_MAP)
    size += NUM_TILES; // Tile resources
    size += NUM_TILES; // Tile numbers
    size += NUM_EDGES; // Edge owners
    size += NUM_NODES; // Node owners
    size += NUM_NODES; // Node building types
    size += NUM_PORTS; // Port resources

    // Players state
    size += n; // Color seating order
    size += (1 + PLAYER_RESOURCES_SIZE + PLAYER_DEVCARDS_SIZE + PLAYER_PLAYED_DEVCARDS_SIZE) * n;

    size
}

pub fn bank_resource_index(resource: u8) -> usize {
    if resource > 4 {
        panic!("Invalid resource index");
    }
    resource as usize
}

pub fn seating_order_slice(num_players: usize) -> std::ops::Range<usize> {
    let slice = PLAYER_STATE_START_INDEX..PLAYER_STATE_START_INDEX + num_players;
    log::debug!(
        "seating_order_slice: num_players={}, PLAYER_STATE_START_INDEX={}, slice={:?}",
        num_players,
        PLAYER_STATE_START_INDEX,
        slice
    );
    slice
}

pub fn actual_victory_points_index(num_players: u8, color: u8) -> usize {
    PLAYER_STATE_START_INDEX + num_players as usize + color as usize * PLAYER_STATE_SIZE
}

pub fn player_hand_slice(num_players: u8, color: u8) -> std::ops::Range<usize> {
    let start = PLAYER_STATE_START_INDEX
        + num_players as usize
        + PLAYER_RESOURCES_OFFSET
        + (color as usize * PLAYER_STATE_SIZE);
    start..start + PLAYER_RESOURCES_SIZE
}

pub fn player_devhand_slice(num_players: u8, color: u8) -> std::ops::Range<usize> {
    let start = PLAYER_STATE_START_INDEX
        + num_players as usize
        + PLAYER_DEVCARDS_OFFSET
        + (color as usize * PLAYER_STATE_SIZE);
    start..start + PLAYER_DEVCARDS_SIZE
}

pub fn player_played_devhand_slice(num_players: u8, color: u8) -> std::ops::Range<usize> {
    let start = PLAYER_STATE_START_INDEX
        + num_players as usize
        + PLAYER_PLAYED_DEVCARDS_OFFSET
        + (color as usize * PLAYER_STATE_SIZE);
    start..start + PLAYER_PLAYED_DEVCARDS_SIZE
}

pub fn get_free_roads_available(vector: &StateVector) -> u8 {
    vector[FREE_ROADS_AVAILABLE_INDEX]
}

// TODO: I'm not sure if it makes more sense to have this in state.rs?
pub fn take_next_dev_card(vector: &mut StateVector) -> Option<u8> {
    let ptr = vector[DEV_BANK_PTR_INDEX] as usize;
    if ptr >= MAX_DEV_CARDS {
        return None;
    }
    let card = vector[DEV_BANK_START_INDEX + ptr];
    vector[DEV_BANK_PTR_INDEX] += 1;
    Some(card)
}

/// This is a compact representation of the omnipotent state of the game.
/// Fairly close to a bitboard, but not quite. Its a vector of integers.
///
/// To create a feature-vector from the perspective of a player,
/// remember to mask hidden information and hot-encode as needed.
///
/// TODO: Is it better to have it already in hot-encoded rep?
/// For now going with compact representation, to allow MCTS rollouts
/// to be faster (without needing to hot-encode). If any workload
/// needs to create samples (e.g. collect RL trajectories), then
/// they'll have to pay the performance of hot-encoding this into
/// a tensor separately.
///
/// TODO: This is not the only Data Structure to do rollouts.
/// We recommend additional caches and aux data structures for
///  faster rollouts. This one is compact optimized for copying.
/// TODO: Accept a seed for deterministic tests
pub fn initialize_state(num_players: u8) -> Vec<u8> {
    log::debug!(
        "initialize_state: num_players={}, PLAYER_STATE_START_INDEX={}",
        num_players,
        PLAYER_STATE_START_INDEX
    );

    let n = num_players as usize;
    let size = get_state_array_size(n);

    log::debug!("initialize_state: size={}, n={}", size, n);

    let mut vector = vec![0; size];

    // Initialize Bank Resources
    for i in BANK_RESOURCE_SLICE {
        vector[i] = MAX_RESOURCE_COUNT;
    }

    // Initialize Bank Development Cards
    // TODO: Shuffle
    let mut listdeck = starting_dev_listdeck();
    listdeck.shuffle(&mut rand::thread_rng());
    vector[DEV_BANK_START_INDEX..DEV_BANK_END_INDEX].copy_from_slice(&listdeck);
    vector[DEV_BANK_PTR_INDEX] = 0;

    // Initialize Game Controls
    vector[CURRENT_TICK_SEAT_INDEX] = 0;
    vector[CURRENT_TURN_SEAT_INDEX] = 0;
    vector[IS_INITIAL_BUILD_PHASE_INDEX] = 1;
    vector[HAS_PLAYED_DEV_CARD] = 0;
    vector[HAS_ROLLED_INDEX] = 0;
    vector[IS_DISCARDING_INDEX] = 0;
    vector[IS_MOVING_ROBBER_INDEX] = 0;
    vector[IS_BUILDING_ROAD_INDEX] = 0;
    vector[FREE_ROADS_AVAILABLE_INDEX] = 0; // Initially no free roads available (road building dev card)

    // Initialize Extra State
    vector[LONGEST_ROAD_PLAYER_INDEX] = u8::MAX;
    vector[LARGEST_ARMY_PLAYER_INDEX] = u8::MAX;
    vector[ROBBER_TILE_INDEX] = 0;

    // Initialize Players
    let mut player_state_start = PLAYER_STATE_START_INDEX;

    // Shuffle player indices
    let mut color_seating_order = COLORS[0..n].iter().map(|&x| x as u8).collect::<Vec<u8>>();
    color_seating_order.shuffle(&mut rand::thread_rng());
    vector[player_state_start..player_state_start + n].copy_from_slice(&color_seating_order);
    player_state_start += n;

    for _ in 0..num_players {
        // Player<i>_Victory_Points (Number <= 12). i is in order of COLORS
        vector[player_state_start] = 0; // victory points

        // Player<i>_<resource>_In_Hand (Number <= 19)
        vector[player_state_start + 1] = 0; // wood
        vector[player_state_start + 2] = 0; // brick
        vector[player_state_start + 3] = 0; // sheep
        vector[player_state_start + 4] = 0; // wheat
        vector[player_state_start + 5] = 0; // ore

        // Player<i>_<devcard>_In_Hand (Number <= 25)
        vector[player_state_start + 6] = 0; // knight
        vector[player_state_start + 7] = 0; // year of plenty
        vector[player_state_start + 8] = 0; // monopoly
        vector[player_state_start + 9] = 0; // road building
        vector[player_state_start + 10] = 0; // victory point

        // Player<i>_<devcard>_Played (Number <= 14)
        vector[player_state_start + 11] = 0; // knight played
        vector[player_state_start + 12] = 0; // year of plenty played
        vector[player_state_start + 13] = 0; // monopoly played
        vector[player_state_start + 14] = 0; // road building played
        player_state_start += PLAYER_STATE_SIZE;
    }

    vector
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::Color;

    #[test]
    fn test_initialize_state_vector() {
        let n: usize = 2;
        let result = get_state_array_size(n);
        assert_eq!(result, 301);
    }

    #[test]
    fn test_initialize_state() {
        let state = initialize_state(2);
        assert_eq!(state.len(), 301);
    }

    #[test]
    fn test_colors_slice() {
        let result = seating_order_slice(4);
        assert_eq!(result, 268..272);
    }

    #[test]
    fn test_indexing() {
        let num_players = 2;
        let result = actual_victory_points_index(num_players, Color::Red as u8);
        assert_eq!(result, 270);

        let result = actual_victory_points_index(num_players, Color::Blue as u8);
        assert_eq!(result, 270 + 15);
    }
}
