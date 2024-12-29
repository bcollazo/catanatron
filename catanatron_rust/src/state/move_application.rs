use std::collections::HashSet;

use rand::Rng;

use crate::{
    deck_slices::*,
    enums::Action,
    map_instance::EdgeId,
    state::Building,
    state_vector::*,
};

use super::State;

impl State {
    pub fn apply_action(&mut self, action: Action) {
        match action {
            Action::BuildSettlement(color, node_id) => {
                self.build_settlement(color, node_id);
            }
            Action::BuildRoad(color, edge_id) => {
                self.build_road(color, edge_id);
            }
            Action::BuildCity(color, node_id) =>{
                self.build_city(color, node_id);
            }
            Action::Roll(color, dice_opt) => {
                self.roll_dice(color, dice_opt);
            }
            Action::Discard(color) => {
                self.discard(color);
            }
            Action::MoveRobber(color, coord, victim_opt) => {
                self.move_robber(color, coord, victim_opt);
            }
            _ => {
                panic!("Action not implemented: {:?}", action);
            }
        }

        println!("Applying action {:?}", action);
    }

    pub fn add_victory_points(&mut self, color: u8, points: u8) {
        let n = self.get_num_players();
        self.vector[actual_victory_points_index(n, color)] += points;
    }

    pub fn advance_turn(&mut self, step_size: i8) {
        // We add an extra num_players to ensure next_index is positive (u8)
        let num_players = self.get_num_players() as i8;
        let next_index =
            ((self.get_current_tick_seat() as i8 + step_size + num_players) % num_players) as u8;

        self.vector[CURRENT_TURN_SEAT_INDEX] = next_index;
    }

    pub fn build_settlement(&mut self, color: u8, node_id: u8) {
        self.buildings
            .insert(node_id, Building::Settlement(color, node_id));
        self.buildings_by_color
            .entry(color)
            .or_default()
            .push(Building::Settlement(color, node_id));

        let is_free = self.is_initial_build_phase();
        if !is_free {
            freqdeck_sub(self.get_mut_player_hand(color), SETTLEMENT_COST);
            freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], SETTLEMENT_COST);
        }

        self.add_victory_points(color, 1);

        if self.is_initial_build_phase() {
            let owned_buildings = self.buildings_by_color.get(&color).unwrap();
            let owned_settlements = owned_buildings
                .iter()
                .filter(|b| matches!(b, Building::Settlement(_, _)))
                .count();

            // If second house, yield resources
            if owned_settlements == 2 {
                let adjacent_tiles = self.map_instance.get_adjacent_tiles(node_id);
                if let Some(adjacent_tiles) = adjacent_tiles {
                    let mut total_resources = [0; 5];
                    for tile in adjacent_tiles {
                        if let Some(resource) = tile.resource {
                            total_resources[resource as usize] += 1;
                        }
                    }

                    let bank = &mut self.vector[BANK_RESOURCE_SLICE];
                    freqdeck_sub(bank, total_resources);

                    let hand = self.get_mut_player_hand(color);
                    freqdeck_add(hand, total_resources);
                }
            }
            // Maintain caches and longest road =====
            //   - connected_components
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
        } else if a_index.is_some() && b_index.is_some() && a_index != b_index {
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

    fn build_city(&mut self, color: u8, node_id: u8) {
        self.buildings.insert(node_id, Building::City(color, node_id));
        let buildings = self.buildings_by_color.entry(color).or_default();
        if let Some (pos) = buildings.iter().position(|b| {
            if let Building::Settlement(_, n) = b {
                *n == node_id
            } else {
                false
            }
        }) {
            buildings.remove(pos);
        }
        freqdeck_sub(self.get_mut_player_hand(color), CITY_COST);
        freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], CITY_COST);
        self.add_victory_points(color, 1);
    }

    fn roll_dice(&mut self, color: u8, dice_opt: Option<(u8, u8)>) {
        self.vector[HAS_ROLLED_INDEX] = 1;
        let (die1, die2) = dice_opt.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            (rng.gen_range(1..=6), rng.gen_range(1..=6))
        });
        let total = die1 + die2;

        if total == 7 {
            let discarders: Vec<bool> = (0..self.get_num_players())
                .map(|c| {
                    let player_hand = self.get_player_hand(c);
                    let total_cards: u8 = player_hand.iter().sum();
                    total_cards > self.config.discard_limit
            })
            .collect();
            
            let should_enter_discard_phase = discarders.iter().any(|&x| x);
            if should_enter_discard_phase {
                if let Some(first_discarder) = discarders.iter().position(|&x| x) {
                    self.vector[CURRENT_TICK_SEAT_INDEX] = first_discarder as u8;
                    self.vector[IS_DISCARDING_INDEX] = 1;
                }
            } else {
                self.vector[IS_MOVING_ROBBER_INDEX] = 1;
                self.vector[CURRENT_TICK_SEAT_INDEX] = color;
            }
        } else {
            // TODO: Yield resources
            self.vector[CURRENT_TICK_SEAT_INDEX] = color;
        }
        // TODO: Set playable_actions???
    }

    fn discard(&mut self, color: u8) {
        todo!();
    }

    fn move_robber(&mut self, color: u8, coordinate: (i8, i8, i8), victim_opt: Option<u8>) {
        self.vector[ROBBER_TILE_INDEX] = self.map_instance
            .get_land_tile(coordinate)
            .unwrap()
            .id;

        if let Some(victim) = victim_opt {
            let total_cards: u8 = self.get_player_hand(victim).iter().sum();

            if total_cards > 0 {
                // Randomly select card to steal
                let mut rng = rand::thread_rng();
                let selected_idx = rng.gen_range(0..total_cards);

                let mut cumsum = 0;
                let mut stolen_resource_idx = 0;
                for (i, &count) in self.get_player_hand(victim).iter().enumerate() {
                    cumsum += count;
                    if selected_idx < cumsum {
                        stolen_resource_idx = i;
                        break;
                    }
                }

                let mut stolen_freqdeck = [0; 5];
                stolen_freqdeck[stolen_resource_idx] = 1;
                freqdeck_sub(self.get_mut_player_hand(victim), stolen_freqdeck);
                freqdeck_add(self.get_mut_player_hand(color), stolen_freqdeck);
            }
        }
        self.vector[IS_MOVING_ROBBER_INDEX] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_settlement_initial_build_phase() {
        let mut state = State::new_base();
        let color = state.get_current_color();
        assert_eq!(state.buildings.get(&0), None);
        assert_eq!(state.board_buildable_ids.len(), 54);
        assert_eq!(state.get_actual_victory_points(color), 0);

        let node_id = 0;
        state.build_settlement(color, node_id);

        assert_eq!(
            state.buildings.get(&node_id),
            Some(&Building::Settlement(color, node_id))
        );
        assert_eq!(state.board_buildable_ids.len(), 50);
        assert_eq!(state.get_actual_victory_points(color), 1);
    }

    #[test]
    fn test_build_settlement_spends_resources() {
        let mut state = State::new_base();
        let color = state.get_current_color();
        assert_eq!(state.buildings.get(&0), None);
        assert_eq!(state.board_buildable_ids.len(), 54);
        assert_eq!(state.get_actual_victory_points(color), 0);

        // Exit initial build phase
        state.vector[IS_INITIAL_BUILD_PHASE_INDEX] = 0;

        freqdeck_add(state.get_mut_player_hand(color), SETTLEMENT_COST);
        let hand_before = state.get_player_hand(color).to_vec();

        let node_id = 0;
        state.build_settlement(color, node_id);

        assert_eq!(
            state.buildings.get(&node_id),
            Some(&Building::Settlement(color, node_id))
        );
        assert_eq!(state.board_buildable_ids.len(), 50);
        assert_eq!(state.get_actual_victory_points(color), 1);
        
        let hand_after = state.get_player_hand(color);
        for i in 0..5 {
            assert_eq!(hand_after[i], hand_before[i] - SETTLEMENT_COST[i]);
        }
    }

    #[test]
    fn test_roll_seven_triggers_discard() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        {
            let hand = state.get_mut_player_hand(color);
            hand[0] = 8; // Give 8 wood cards
        }

        state.roll_dice(color, Some((4, 3)));

        assert_eq!(state.vector[HAS_ROLLED_INDEX], 1);
        assert_eq!(state.vector[IS_DISCARDING_INDEX], 1);
        assert_eq!(state.vector[CURRENT_TICK_SEAT_INDEX], color);
        assert_eq!(state.vector[IS_MOVING_ROBBER_INDEX], 0);
    }

    #[test]
    fn test_roll_seven_no_discard_needed() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        state.roll_dice(color, Some((4, 3)));

        assert_eq!(state.vector[HAS_ROLLED_INDEX], 1);
        assert_eq!(state.vector[IS_DISCARDING_INDEX], 0);
        assert_eq!(state.vector[CURRENT_TICK_SEAT_INDEX], color);
        assert_eq!(state.vector[IS_MOVING_ROBBER_INDEX], 1);
    }

    #[test]
    fn test_roll_tracks_has_rolled() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        assert_eq!(state.vector[HAS_ROLLED_INDEX], 0);
        state.roll_dice(color, Some((2, 3)));
        assert_eq!(state.vector[HAS_ROLLED_INDEX], 1);
    }

    #[test]
    fn test_second_settlement_yields_resources() {
        let mut state = State::new_base();
        let color = state.get_current_color();
        let first_node = 0;
        let bank_before = state.vector[BANK_RESOURCE_SLICE].to_vec();
        let hand_before = state.get_player_hand(color).to_vec();

        state.build_settlement(color, first_node);

        assert_eq!(state.get_player_hand(color), hand_before);
        assert_eq!(state.vector[BANK_RESOURCE_SLICE], bank_before);

        let second_node = 3;
        let bank_before = state.vector[BANK_RESOURCE_SLICE].to_vec();
        let hand_before = state.get_player_hand(color).to_vec();

        state.build_settlement(color, second_node);

        assert_ne!(state.get_player_hand(color), hand_before);
        assert_ne!(state.vector[BANK_RESOURCE_SLICE], bank_before);

        for i in 0..5 {
            let bank_diff = bank_before[i] - state.vector[BANK_RESOURCE_SLICE][i];
            let hand_diff = state.get_player_hand(color)[i] - hand_before[i];
            assert_eq!(bank_diff, hand_diff);
        }
    }
}
