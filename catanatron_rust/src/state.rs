use std::{collections::HashSet, rc::Rc};

use crate::{
    enums::{ActionPrompt, GameConfiguration},
    map_instance::{MapInstance, NodeId},
    state_vector::{
        actual_victory_points_index, initialize_state, seating_order_slice, StateVector,
        CURRENT_TICK_SEAT_INDEX, IS_DISCARDING_INDEX, IS_INITIAL_BUILD_PHASE_INDEX,
        IS_MOVING_ROBBER_INDEX,
    },
};

#[derive(Debug)]
pub(crate) struct State {
    // These two are immutable
    config: Rc<GameConfiguration>,
    map_instance: Rc<MapInstance>,

    // This is mutable
    vector: StateVector,

    // These are caches for speeding up game state calculations
    board_buildable_ids: HashSet<NodeId>,
    longest_road_color: Option<u8>,
    longest_road_length: u8,
}

impl State {
    pub fn new(config: Rc<GameConfiguration>, map_instance: Rc<MapInstance>) -> Self {
        let vector = initialize_state(config.num_players);

        let board_buildable_ids = map_instance.land_nodes().clone();
        let longest_road_color = None;
        let longest_road_length = 0;

        Self {
            config,
            map_instance,
            vector,
            board_buildable_ids,
            longest_road_color,
            longest_road_length,
        }
    }

    // ===== Config Getters =====
    pub fn get_num_players(&self) -> u8 {
        self.config.num_players
    }
    pub fn get_num_players_usize(&self) -> usize {
        self.config.num_players as usize
    }

    // ===== Getters =====
    pub fn is_initial_build_phase(&self) -> bool {
        self.vector[IS_INITIAL_BUILD_PHASE_INDEX] == 1
    }

    pub fn is_moving_robber(&self) -> bool {
        self.vector[IS_MOVING_ROBBER_INDEX] == 1
    }

    pub fn is_discarding(&self) -> bool {
        self.vector[IS_DISCARDING_INDEX] == 1
    }

    pub fn get_current_color(&self) -> u8 {
        let seating_order = self.get_seating_order();
        let current_tick_seat = self.vector[CURRENT_TICK_SEAT_INDEX];
        seating_order[current_tick_seat as usize]
    }

    /// Returns a slice of Colors in the order of seating
    /// e.g. [2, 1, 0, 3] if Orange goes first, then Blue, then Red, and then White
    pub fn get_seating_order(&self) -> &[u8] {
        &self.vector[seating_order_slice(self.config.num_players as usize)]
    }

    pub fn get_action_prompt(&self) -> ActionPrompt {
        if self.is_initial_build_phase() {
            let num_things_built = 0;
            if num_things_built == 2 * self.config.num_players {
                return ActionPrompt::PlayTurn;
            } else if num_things_built % 2 == 0 {
                return ActionPrompt::BuildInitialSettlement;
            } else {
                return ActionPrompt::BuildInitialRoad;
            }
        } else if self.is_moving_robber() {
            return ActionPrompt::MoveRobber;
        } else if self.is_discarding() {
            return ActionPrompt::Discard;
        } // TODO: Implement Trading Prompts (DecideTrade, DecideAcceptees)
        ActionPrompt::PlayTurn
    }

    pub fn winner(&self) -> Option<u8> {
        let current_color = self.get_current_color();

        let actual_victory_points =
            self.vector[actual_victory_points_index(self.config.num_players, current_color)];
        if actual_victory_points >= self.config.vps_to_win {
            return Some(current_color);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{enums::MapType, global_state::GlobalState};

    #[test]
    fn test_state_creation() {
        let global_state = GlobalState::new();
        let config = GameConfiguration {
            dicard_limit: 7,
            vps_to_win: 10,
            map_type: MapType::Base,
            num_players: 4,
            max_turns: 100,
        };
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );
        let state = State::new(Rc::new(config), Rc::new(map_instance));

        assert_eq!(state.longest_road_color, None);
    }
}
