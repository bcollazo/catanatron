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
                .or_default()
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
            freqdeck_sub(self.get_mut_player_hand(color), SETTLEMENT_COST);
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

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::{
        enums::{GameConfiguration, MapType},
        global_state::GlobalState,
        map_instance::MapInstance,
    };

    fn setup_state() -> State {
        let global_state = GlobalState::new();
        let config = GameConfiguration {
            dicard_limit: 7,
            vps_to_win: 10,
            map_type: MapType::Base,
            num_players: 2,
            max_turns: 10,
        };
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );
        State::new(Rc::new(config), Rc::new(map_instance))
    }

    #[test]
    fn test_build_settlement() {
        let mut state = setup_state();
        let color = state.get_current_color();
        assert_eq!(state.buildings.get(&0), None);
        assert_eq!(state.board_buildable_ids.len(), 54);
        assert_eq!(state.get_actual_victory_points(color), 0);

        let node_id = 0;
        state.build_settlement(color, node_id);

        assert_eq!(
            state.buildings.get(&node_id),
            Some(&Building::Settlement(color))
        );
        assert_eq!(state.board_buildable_ids.len(), 50);
        assert_eq!(state.get_actual_victory_points(color), 1);
    }

    // TODO: Assert build_settlement spends player resources
}
