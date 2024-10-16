use crate::decks::starting_dev_listdeck;
use rand::seq::SliceRandom;

use crate::enums::COLORS;

/// This is in theory not needed since we use a vector and we can
/// .push() to it. But since we made it, leaving in here in case
/// we want to switch to an array implementation and it serves
/// as documentation of the state vector.
pub fn get_state_array_size(_num_players: u8) -> usize {
    // TODO: Is configuration part of state?
    // TODO: Hardcoded for BASE_MAP
    let n = _num_players as usize;
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
pub fn initialize_state() -> Vec<u8> {
    let num_players = 2;
    let size = get_state_array_size(num_players);
    let mut vector = vec![0; size];
    // Initialize Bank
    vector[0] = 19;
    vector[1] = 19;
    vector[2] = 19;
    vector[3] = 19;
    vector[4] = 19;
    // Initialize Bank Development Cards
    let listdeck = starting_dev_listdeck();
    vector[5..30].copy_from_slice(&listdeck);
    // Initialize Game Controls
    vector[30] = 0; // Current_Player_Index
    vector[31] = 0; // Current_Turn_Index
    vector[32] = 1; // Is_Initial_Build_Phase
    vector[33] = 0; // Has_Played_Development_Card
    vector[34] = 0; // Has_Rolled
    vector[35] = 0; // Is_Discarding
    vector[36] = 0; // Is_Moving_Robber
    vector[37] = 0; // Is_Building_Road
    vector[38] = 2; // Free_Roads_Available

    // Extra (u8::MAX is used to indicate no player)
    vector[39] = u8::MAX; // Longest_Road_Player_Index
    vector[40] = u8::MAX; // Largest_Army_Player_Index

    // Board
    // TODO: Generate map from template
    // vector[41] = 0;
    // size += 1; // Robber_Tile (Tile Index < num_tiles)
    // size += num_tiles; // Tile<i>_Resource (Resource Index <= 5)
    // size += num_tiles; // Tile<i>_Number (Number <= 12)
    // size += num_edges; // Edge<i>_Owner (Player Index | -1 < n + 1)
    // size += num_nodes; // Node<i>_Owner (Player Index | -1 < n + 1)
    // size += num_nodes; // Node<i>_Settlement/City (1=Settlement, 2=City, 0=Nothing)
    // size += num_ports; // Port<i>_Resource (Resource Index <= 5)

    // Initialize Players
    // Shuffle player indices
    let mut player_indices = COLORS.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    player_indices.shuffle(&mut rand::thread_rng());
    vector[267..267 + player_indices.len()].copy_from_slice(&player_indices);
    let mut player_state_start = 267 + player_indices.len();
    for _ in 0..num_players {
        // Player<i>_Victory_Points (Number <= 12)
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

    #[test]
    fn test_initialize_state_vector() {
        let n: usize = 2;
        let result = get_state_array_size(n as u8);
        assert_eq!(result, 301);
    }

    #[test]
    fn test_initialize_state() {
        initialize_state();
    }
}
