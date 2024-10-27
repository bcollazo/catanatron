use std::collections::HashSet;

use crate::{
    deck_slices::{freqdeck_add, freqdeck_sub, ROAD_COST, SETTLEMENT_COST},
    enums::Action,
    map_instance::EdgeId,
    state::Building,
    state_vector::{
        actual_victory_points_index, BANK_RESOURCE_SLICE, CURRENT_TICK_SEAT_INDEX,
        CURRENT_TURN_SEAT_INDEX, IS_INITIAL_BUILD_PHASE_INDEX,
    },
};

use super::State;

impl State {
    pub fn add_victory_points(&mut self, color: u8, points: u8) {
        let n = self.get_num_players();
        self.vector[actual_victory_points_index(n, color)] += points;
    }

    pub fn advance_turn(&mut self, step_size: i8) {
        // We add an extra num_players to ensure next_index is positive (u8)
        let num_players = self.get_num_players() as i8;
        let next_index =
            ((self.get_current_tick_seat() as i8 + step_size + num_players) % num_players) as u8;
        self.vector[CURRENT_TICK_SEAT_INDEX] = next_index;
        self.vector[CURRENT_TURN_SEAT_INDEX] = next_index;
    }

    pub fn build_settlement(&mut self, color: u8, node_id: u8) {
        self.buildings.insert(node_id, Building::Settlement(color));

        let is_free = self.is_initial_build_phase();
        if !is_free {
            freqdeck_sub(self.get_mut_player_hand(color), SETTLEMENT_COST);
            freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], SETTLEMENT_COST);
        }

        self.add_victory_points(color, 1);

        // TODO: If second house, yield resources

        // Maintain caches and longest road =====
        //   - connected_components
        if self.is_initial_build_phase() {
            let component = HashSet::from([node_id]);
            self.connected_components
                .entry(color)
                .or_default()
                .push(component);
        } else {
            // TODO: Mantain connected_components
            // TODO: Mantain longest_road_color and longest_road_length (maybe swapping vps)
            todo!();
        }
        // - board_buildable_ids
        self.board_buildable_ids.remove(&node_id);
        for neighbor_id in self.map_instance.get_neighbor_nodes(node_id) {
            self.board_buildable_ids.remove(&neighbor_id);
        }
    }

    fn build_road(&mut self, color: u8, edge_id: EdgeId) {
        let inverted_edge = (edge_id.1, edge_id.0);
        self.roads.insert(edge_id, color);
        self.roads.insert(inverted_edge, color);

        let is_initial_build_phase = self.is_initial_build_phase();
        let is_free = is_initial_build_phase || self.is_road_building();
        if !is_free {
            freqdeck_sub(self.get_mut_player_hand(color), ROAD_COST);
            freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], ROAD_COST);
        }

        if is_initial_build_phase {
            let num_settlements = self.buildings.len();
            let num_players = self.config.num_players as usize;
            let going_forward = num_settlements < num_players;
            let at_midpoint = num_settlements == num_players;

            if going_forward {
                self.advance_turn(1);
            } else if at_midpoint {
                // do nothing, generate prompt should take care
            } else if num_settlements == 2 * num_players {
                // just change prompt without advancing turn (since last to place is first to roll)
                self.vector[IS_INITIAL_BUILD_PHASE_INDEX] = 0;
            } else {
                self.advance_turn(-1);
            }
        }

        // Maintain caches and longest road =====
        // Extend or merge components
        let (a, b) = edge_id;
        let a_index = self.get_connected_component_index(color, a);
        let b_index = self.get_connected_component_index(color, b);
        if a_index.is_none() && !self.is_enemy_node(color, a) {
            // There has to be a component from b (since roads can only be built in a connected fashion)
            let component = self
                .connected_components
                .get_mut(&color)
                .unwrap()
                .get_mut(b_index.unwrap())
                .unwrap();
            component.insert(a); // extend said component by 1 more node
        } else if b_index.is_none() && !self.is_enemy_node(color, b) {
            // There has to be a component from a (since roads can only be built in a connected fashion)
            let component = self
                .connected_components
                .get_mut(&color)
                .unwrap()
                .get_mut(a_index.unwrap())
                .unwrap();
            component.insert(b); // extend said component by 1 more node
        } else if !a_index.is_none() && !b_index.is_none() && a_index != b_index {
            // Merge components into one and delete the other
            let a_component = self
                .connected_components
                .get_mut(&color)
                .unwrap()
                .remove(a_index.unwrap());
            let b_component = self
                .connected_components
                .get_mut(&color)
                .unwrap()
                .remove(b_index.unwrap());
            let mut new_component = a_component.clone();
            new_component.extend(b_component);
            self.connected_components
                .get_mut(&color)
                .unwrap()
                .push(new_component);
        } else {
            // In this case, a_index == b_index, which means that the edge
            // is already part of one component. No actions needed.
        }

        // TODO: Return previous road
    }

    pub fn apply_action(&mut self, action: Action) {
        match action {
            Action::BuildSettlement(color, node_id) => {
                self.build_settlement(color, node_id);
            }
            Action::BuildRoad(color, edge_id) => {
                self.build_road(color, edge_id);
            }
            _ => {
                panic!("Action not implemented: {:?}", action);
            }
        }

        println!("Applying action {:?}", action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_settlement() {
        let mut state = State::new_base();
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
