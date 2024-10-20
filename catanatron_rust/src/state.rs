use std::collections::HashSet;

use crate::{
    enums::GameConfiguration,
    map_instance::{MapInstance, NodeId},
    state_vector::{initialize_state, StateVector},
};

// Helpful Cache of information for algorithms
struct Board {
    board_buildable_ids: HashSet<NodeId>,
    longest_road_color: Option<u8>,
    longest_road_length: u8,
}

impl Board {
    pub fn new(map_instance: &MapInstance) -> Self {
        let board_buildable_ids = map_instance.land_nodes().clone();

        // TODO: Keep filling me!
        Self {
            board_buildable_ids,
            longest_road_color: None,
            longest_road_length: 0,
        }
    }
}

struct State {
    // These two are immutable
    config: GameConfiguration,
    map_instance: MapInstance,

    // This is mutable
    vector: StateVector,

    // These are caches for speeding up game state calculations
    board: Board,
}

impl State {
    pub fn new(config: GameConfiguration, map_instance: MapInstance) -> Self {
        let vector = initialize_state(config.num_players);
        let board = Board::new(&map_instance);
        Self {
            config,
            map_instance,
            vector,
            board,
        }
    }
}
