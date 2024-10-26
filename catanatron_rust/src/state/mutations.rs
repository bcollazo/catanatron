use std::collections::HashSet;

use crate::{
    deck_slices::{freqdeck_add, freqdeck_sub, SETTLEMENT_COST},
    enums::Action,
    state::Building,
    state_vector::{actual_victory_points_index, BANK_RESOURCE_SLICE},
};

use super::State;

impl State {
    pub fn build_settlement(&mut self, color: u8, node_id: u8) {
        println!("Building settlement {:?} {:?}", color, node_id);
        self.buildings.insert(node_id, Building::Settlement(color));

        // Maintain caches
        //   - connected_components
        if self.is_initial_build_phase() {
            let component = HashSet::from([node_id]);
            self.connected_components
                .entry(color)
                .or_insert(Vec::new())
                .push(component);
        } else {
            todo!();
        }
        // - board_buildable_ids
        self.board_buildable_ids.remove(&node_id);
        for neighbor_id in self.map_instance.get_neighbor_nodes(node_id) {
            self.board_buildable_ids.remove(&neighbor_id);
        }

        let n = self.get_num_players();
        self.vector[actual_victory_points_index(n, color)] += 1;

        let is_initial_build_phase = self.is_initial_build_phase();
        if !is_initial_build_phase {
            freqdeck_sub(&mut self.get_mut_player_hand(color), SETTLEMENT_COST);
            freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], SETTLEMENT_COST);
        }
    }

    pub fn apply_action(&mut self, action: Action) {
        match action {
            Action::BuildSettlement(color, node_id) => {
                self.build_settlement(color, node_id);
            }
            _ => {
                println!("Action not implemented: {:?}", action);
            }
        }

        println!("Applying action {:?}", action);
    }
}
