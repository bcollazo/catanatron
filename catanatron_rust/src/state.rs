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
pub fn initialize_state_vector(_num_players: u8) -> Vec<u8> {
    // TODO: Is configuration part of state?
    // TODO: Hardcoded for BASE_MAP
    let n = _num_players as usize;
    let num_nodes = 54;
    let num_edges = 72;
    let num_tiles = 19;
    let num_ports = 9;

    let mut size: usize = 0;
    // Bank
    size += 5; // Bank Resources (Number <= 19)
    size += 1; // Bank Development Cards (Number <= 25)

    // Game Controls
    size += n; // Color_Seating_Order (Player Index < n)
    size += 1; // Current_Player_Index (Player Index < n)
    size += 1; // Current_Turn_Index (Player Index < n)
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

    // Players
    let mut player_state_size: usize = 0;
    player_state_size += 1; // Player<i>_Victory_Points (Number <= 12)
    player_state_size += 5; // Player<i>_<resource>_In_Hand (Number <= 19)
    player_state_size += 5; // Player<i>_<devcard>_In_Hand (Number <= 25)
    player_state_size += 5; // Player<i>_<devcard>_Played (Number <= 14)

    // This is redundant information (since one can figure out from the board state)
    // player_state_size += 5; // Player<i>_<resource>_Roads_Left
    // player_state_size += 5; // Player<i>_<resource>_Settlements_Left
    // player_state_size += 5; // Player<i>_<resource>_Cities_Left
    size += player_state_size * n;

    // Board
    size += 1; // Robber_Tile (Tile Index < num_tiles)
    size += num_tiles; // Tile<i>_Resource (Resource Index <= 5)
    size += num_tiles; // Tile<i>_Number (Number <= 12)
    size += num_edges; // Edge<i>_Owner (Player Index | -1 < n + 1)
    size += num_nodes; // Node<i>_Owner (Player Index | -1 < n + 1)
    size += num_nodes; // Node<i>_Settlement/City (1=Settlement, 2=City, 0=Nothing)
    size += num_ports; // Port<i>_Resource (Resource Index <= 5)

    // TODO: This is not the only Data Structure to do rollouts.
    //  We recommend additional caches and aux data structures for
    //  faster rollouts. This one is compact optimized for copying.
    let vector = vec![0u8; size];
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_state_vector() {
        let n: usize = 2;
        let result = initialize_state_vector(n as u8);
        assert_eq!(result.len(), 278);
    }
}
