use crate::{decks::starting_dev_listdeck, map_instance::MapInstance};
use rand::seq::SliceRandom;

use crate::enums::COLORS;

pub type StateVector = Vec<u8>;

/// This is in theory not needed since we use a vector and we can
/// .push() to it. But since we made it, leaving in here in case
/// we want to switch to an array implementation and it serves
/// as documentation of the state vector.
pub fn get_state_array_size(num_players: usize) -> usize {
    // TODO: Is configuration part of state?
    // TODO: Hardcoded for BASE_MAP
    let n = num_players;
    let num_nodes = 54;
    let num_edges = 72;
    let num_tiles = 19;
    let num_ports = 9;

    let mut size: usize = 0;
    // Trying to have as most fixed-size vector first as possible
    //  so that we can understand/debug all configurations similarly.
    // Bank
    size += 5; // Bank Resources (Number <= 19)
    size += 25; // Bank Development Cards (DevCard Index <= 5)

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

    // Board (dynamically sized based on map template; 228 for BASE_MAP)
    size += 1; // Robber_Tile (Tile Index < num_tiles)
    size += num_tiles; // Tile<i>_Resource (Resource Index <= 5)
    size += num_tiles; // Tile<i>_Number (Number <= 12) 39
    size += num_edges; // Edge<i>_Owner (Player Index | -1 < n + 1)
    size += num_nodes; // Node<i>_Owner (Player Index | -1 < n + 1)
    size += num_nodes; // Node<i>_Settlement/City (1=Settlement, 2=City, 0=Nothing)
    size += num_ports; // Port<i>_Resource (Resource Index <= 5)

    // Players (dynamically sized based on number of players = 1 + 15 x n)
    size += n; // Color_Seating_Order (Player Index < n)
    let mut player_state_size: usize = 0;
    player_state_size += 1; // Player<i>_Victory_Points (Number <= 12)
    player_state_size += 5; // Player<i>_<resource>_In_Hand (Number <= 19)
    player_state_size += 5; // Player<i>_<devcard>_In_Hand (Number <= 25)
    player_state_size += 4; // Player<i>_<devcard>_Played (Number <= 14) VictoryPoint can't be played

    // This is redundant information (since one can figure out from the board state)
    // player_state_size += 5; // Player<i>_<resource>_Roads_Left
    // player_state_size += 5; // Player<i>_<resource>_Settlements_Left
    // player_state_size += 5; // Player<i>_<resource>_Cities_Left
    size += player_state_size * n;

    size
}

pub fn bank_resource_index(resource: u8) -> usize {
    if resource > 4 {
        panic!("Invalid resource index");
    }
    resource as usize
}
pub const BANK_RESOURCE_SLICE: std::ops::Range<usize> = 0..5;
const PLAYER_STATE_START_INDEX: usize = 268;
pub fn seating_order_slice(num_players: usize) -> std::ops::Range<usize> {
    PLAYER_STATE_START_INDEX..PLAYER_STATE_START_INDEX + num_players
}
pub fn actual_victory_points_index(num_players: u8, color: u8) -> usize {
    PLAYER_STATE_START_INDEX + num_players as usize + color as usize * 15
}
pub const DEV_BANK_PTR_INDEX: usize = 30;
pub const CURRENT_TICK_SEAT_INDEX: usize = 31;
pub const CURRENT_TURN_SEAT_INDEX: usize = 32;
pub const IS_INITIAL_BUILD_PHASE_INDEX: usize = 33;
pub const HAS_PLAYED_DEV_CARD: usize = 34;
pub const HAS_ROLLED_INDEX: usize = 35;
pub const IS_DISCARDING_INDEX: usize = 36;
pub const IS_MOVING_ROBBER_INDEX: usize = 37;

pub const ROBBER_TILE_INDEX: usize = 42;
pub fn player_hand_slice(color: u8) -> std::ops::Range<usize> {
    let start = PLAYER_STATE_START_INDEX + 1 + (color as usize * 15);
    start..start + 5
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
    let n = num_players as usize;

    let size = get_state_array_size(n);
    let mut vector = vec![0; size];
    // Initialize Bank
    vector[0] = 19;
    vector[1] = 19;
    vector[2] = 19;
    vector[3] = 19;
    vector[4] = 19;
    // Initialize Bank Development Cards
    // TODO: Shuffle
    let mut listdeck = starting_dev_listdeck();
    listdeck.shuffle(&mut rand::thread_rng());
    vector[5..30].copy_from_slice(&listdeck);
    // May be worth storing a pointer to the top of the deck
    vector[DEV_BANK_PTR_INDEX] = 0;
    // Initialize Game Controls
    vector[CURRENT_TICK_SEAT_INDEX] = 0;
    vector[CURRENT_TURN_SEAT_INDEX] = 0;
    vector[IS_INITIAL_BUILD_PHASE_INDEX] = 1; // Is_Initial_Build_Phase
    vector[HAS_PLAYED_DEV_CARD] = 0; // Has_Played_Development_Card
    vector[HAS_ROLLED_INDEX] = 0; // Has_Rolled
    vector[IS_DISCARDING_INDEX] = 0; // Is_Discarding
    vector[IS_MOVING_ROBBER_INDEX] = 0; // Is_Moving_Robber
    vector[38] = 0; // Is_Building_Road
    vector[39] = 2; // Free_Roads_Available

    // Extra (u8::MAX is used to indicate no player)
    vector[40] = u8::MAX; // Longest_Road_Player_Index
    vector[41] = u8::MAX; // Largest_Army_Player_Index

    // Board
    // TODO: Generate map from template
    vector[ROBBER_TILE_INDEX] = 0; // Robber_Tile

    let mut player_state_start = PLAYER_STATE_START_INDEX;

    // Initialize Players
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
        player_state_start += 15;
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
