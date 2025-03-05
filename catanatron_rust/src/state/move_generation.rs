use crate::deck_slices::{freqdeck_contains, CITY_COST, ROAD_COST, SETTLEMENT_COST};
use crate::enums::{Action, ActionPrompt, DevCard};
use crate::state::State;
use crate::state_vector::get_free_roads_available;
use std::collections::HashSet;

use super::Building;

const TOTAL_ROADS_PER_PLAYER: u8 = 15;
const TOTAL_CITIES_PER_PLAYER: u8 = 4;

impl State {
    pub fn generate_playable_actions(&self) -> Vec<Action> {
        let current_color = self.get_current_color();
        let action_prompt = self.get_action_prompt();
        match action_prompt {
            ActionPrompt::BuildInitialSettlement => {
                self.settlement_possibilities(current_color, true)
            }
            ActionPrompt::BuildInitialRoad => self.initial_road_possibilities(current_color),
            ActionPrompt::MoveRobber => self.robber_possibilities(current_color),
            ActionPrompt::PlayTurn => self.play_turn_possibilities(current_color),
            ActionPrompt::Discard => self.discard_possibilities(current_color),
            ActionPrompt::DecideTrade => todo!("generate_playable_actions for Decide trade"),
            ActionPrompt::DecideAcceptees => {
                todo!("generate_playable_actions for Decide acceptees")
            }
        }
    }

    pub fn settlement_possibilities(&self, color: u8, is_initial_build_phase: bool) -> Vec<Action> {
        if is_initial_build_phase {
            self.board_buildable_ids
                .iter()
                .map(|node_id| Action::BuildSettlement {
                    color,
                    node_id: *node_id,
                })
                .collect()
        } else {
            let has_resources = freqdeck_contains(self.get_player_hand(color), SETTLEMENT_COST);
            let settlements_used = self.get_settlements(color).len();
            let has_settlements_available = settlements_used < 5;

            if has_resources && has_settlements_available {
                self.buildable_node_ids(color)
                    .into_iter()
                    .map(|node_id| Action::BuildSettlement { color, node_id })
                    .collect()
            } else {
                vec![]
            }
        }
    }

    pub fn initial_road_possibilities(&self, color: u8) -> Vec<Action> {
        let last_settlement_building = self.buildings_by_color[&color].last().unwrap();
        let last_node_id = match last_settlement_building {
            Building::Settlement(_, node_id) => *node_id,
            _ => panic!("Invalid building type"),
        };

        self.board_buildable_edges(color)
            .iter()
            .filter(|edge_id| self.edge_contains(**edge_id, last_node_id))
            .map(|edge_id| Action::BuildRoad {
                color,
                edge_id: *edge_id,
            })
            .collect()
    }

    pub fn road_possibilities(&self, color: u8, is_free: bool) -> Vec<Action> {
        let has_roads_available = TOTAL_ROADS_PER_PLAYER - self.roads_by_color[color as usize];
        if has_roads_available == 0 {
            return vec![];
        }

        if is_free || freqdeck_contains(self.get_player_hand(color), ROAD_COST) {
            self.board_buildable_edges(color)
                .iter()
                .map(|edge_id| Action::BuildRoad {
                    color,
                    edge_id: *edge_id,
                })
                .collect()
        } else {
            vec![]
        }
    }

    pub fn city_possibilities(&self, color: u8) -> Vec<Action> {
        let has_money = freqdeck_contains(self.get_player_hand(color), CITY_COST);
        if !has_money {
            return vec![];
        }

        let has_cities_available = self.get_cities(color).len() < TOTAL_CITIES_PER_PLAYER as usize;
        if !has_cities_available {
            return vec![];
        }

        self.get_settlements(color)
            .iter()
            .map(|building| match building {
                Building::Settlement(color, node_id) => Action::BuildCity {
                    color: *color,
                    node_id: *node_id,
                },
                _ => panic!("Invalid building type"),
            })
            .collect()
    }

    pub fn year_of_plenty_possibilities(&self, color: u8) -> Vec<Action> {
        let bank_resources = self.get_bank_resources();
        let mut actions = Vec::new();

        // First try all same-resource pairs
        for (resource, &count) in bank_resources.iter().enumerate() {
            if count >= 2 {
                actions.push(Action::PlayYearOfPlenty {
                    color,
                    resources: (resource as u8, Some(resource as u8)),
                });
            }
        }

        // Then try different-resource pairs
        for (resource1, &count1) in bank_resources.iter().enumerate() {
            if count1 > 0 {
                for (resource2, &count2) in bank_resources.iter().enumerate().skip(resource1 + 1) {
                    if count2 > 0 {
                        actions.push(Action::PlayYearOfPlenty {
                            color,
                            resources: (resource1 as u8, Some(resource2 as u8)),
                        });
                    }
                }
            }
        }

        // If no two-resource actions possible, try single resources
        if actions.is_empty() {
            for (resource, &count) in bank_resources.iter().enumerate() {
                if count > 0 {
                    actions.push(Action::PlayYearOfPlenty {
                        color,
                        resources: (resource as u8, None),
                    });
                    break; // Only need first available resource
                }
            }
        }

        actions
    }

    pub fn play_turn_possibilities(&self, color: u8) -> Vec<Action> {
        if self.is_road_building() {
            if get_free_roads_available(&self.vector) == 0 {
                return vec![Action::EndTurn { color }]; // Always provide EndTurn
            }
            return self.road_possibilities(color, true);
        } else if !self.current_player_rolled() {
            let mut actions = vec![Action::Roll {
                color,
                dice_opt: None,
            }];
            if self.can_play_dev(DevCard::Knight as u8) {
                actions.push(Action::PlayKnight { color });
            }
            return actions;
        }

        let mut actions = vec![Action::EndTurn { color }];

        // Add all possible actions
        actions.extend(self.settlement_possibilities(color, false));
        actions.extend(self.road_possibilities(color, false));
        actions.extend(self.city_possibilities(color));

        if self.can_play_dev(DevCard::Knight as u8) {
            actions.push(Action::PlayKnight { color });
        }
        if self.can_play_dev(DevCard::YearOfPlenty as u8) {
            actions.extend(self.year_of_plenty_possibilities(color));
        }
        if self.can_play_dev(DevCard::Monopoly as u8) {
            for resource in 0..5 {
                actions.push(Action::PlayMonopoly { color, resource });
            }
        }
        if self.can_play_dev(DevCard::RoadBuilding as u8) {
            // TODO: What if user has no roads left? or is completely blocked?
            actions.push(Action::PlayRoadBuilding { color });
        }

        // Add maritime trade possibilities
        actions.extend(self.maritime_trade_possibilities(color));

        // TODO: Domestic trading is temporarily disabled to reduce the state space explosion
        // This simplification allows us to first build a superhuman AI player without
        // the complexity of domestic trading.

        actions
    }

    fn calculate_port_rates(&self, color: u8) -> [u8; 5] {
        let mut port_rates = [4; 5]; // Default 4:1 rate for all resources

        let Some(player_buildings) = self.buildings_by_color.get(&color) else {
            return port_rates;
        };

        // For each player building, check if it's on a port and update rates
        for building in player_buildings {
            let node_id = match building {
                Building::Settlement(_, id) | Building::City(_, id) => id,
            };

            if let Some(&port_resource) = self.map_instance.get_port_nodes().get(node_id) {
                match port_resource {
                    Some(resource) => port_rates[resource as usize] = 2,
                    None => port_rates
                        .iter_mut()
                        .for_each(|rate| *rate = (*rate).min(3)),
                }
            }
        }

        port_rates
    }

    pub fn maritime_trade_possibilities(&self, color: u8) -> Vec<Action> {
        let hand = self.get_player_hand(color);
        let bank = self.get_bank_resources();
        let port_rates = self.calculate_port_rates(color);

        hand.iter()
            .enumerate()
            .flat_map(|(give_idx, &give_count)| {
                let rate = port_rates[give_idx];
                if give_count >= rate {
                    (0..5)
                        .filter(|&take_idx| {
                            // Ensure bank has enough resources and it's a different resource
                            take_idx != give_idx && bank[take_idx] > 0
                        })
                        .map(|take_idx| Action::MaritimeTrade {
                            color,
                            give: give_idx as u8,
                            take: take_idx as u8,
                            ratio: rate,
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    pub fn robber_possibilities(&self, color: u8) -> Vec<Action> {
        let mut actions = vec![];
        let current_robber_tile = self.get_robber_tile();

        for (coordinate, tile) in self.map_instance.get_land_tiles() {
            // Skip current robber location
            if tile.id == current_robber_tile {
                continue;
            }

            // Find players to steal from at this tile
            let mut victims = HashSet::new();
            for node_id in tile.hexagon.nodes.values() {
                if let Some(building) = self.buildings.get(node_id) {
                    match building {
                        Building::Settlement(victim_color, _) | Building::City(victim_color, _) => {
                            // Can't steal from yourself and victim must have resources
                            if *victim_color != color
                                && self.get_player_hand(*victim_color).iter().sum::<u8>() > 0
                            {
                                victims.insert(*victim_color);
                            }
                        }
                    }
                }
            }

            if victims.is_empty() {
                actions.push(Action::MoveRobber {
                    color,
                    coordinate: *coordinate,
                    victim_opt: None,
                });
            } else {
                for victim in victims {
                    actions.push(Action::MoveRobber {
                        color,
                        coordinate: *coordinate,
                        victim_opt: Some(victim),
                    });
                }
            }
        }

        actions
    }

    pub fn discard_possibilities(&self, color: u8) -> Vec<Action> {
        let hand = self.get_player_hand(color);
        let total_cards: u8 = hand.iter().sum();

        // If player has 7 or fewer cards, they can just end their turn
        if total_cards <= 7 {
            return vec![Action::EndTurn { color }];
        }

        // For now, just generate a single discard action
        // This is to prevent state space explosion
        vec![Action::Discard { color }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::Resource;

    fn find_port_node_by_type(state: &State, resource: Option<Resource>) -> Option<u8> {
        state
            .map_instance
            .get_port_nodes()
            .iter()
            .find_map(|(&node_id, &port_resource)| {
                if port_resource == resource {
                    Some(node_id)
                } else {
                    None
                }
            })
    }

    #[test]
    fn test_move_generation() {
        let state = State::new_base();
        let actions = state.generate_playable_actions();
        assert_eq!(actions.len(), 54);
        assert!(matches!(actions[0], Action::BuildSettlement { .. }));
    }

    #[test]
    fn test_settlement_possibilities() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Test initial build phase
        let initial_build_phase_actions = state.settlement_possibilities(color, true);
        assert_eq!(initial_build_phase_actions.len(), 54);

        // Give player resources
        let hand = state.get_mut_player_hand(color);
        for resource in hand.iter_mut() {
            *resource = 5;
        }

        let action = Action::BuildSettlement { color, node_id: 0 };
        state.apply_action(action);

        let action = Action::BuildRoad {
            color,
            edge_id: (0, 1),
        };
        state.apply_action(action);

        let actions = state.settlement_possibilities(0, false);
        assert_eq!(actions.len(), 0);

        let action = Action::BuildRoad {
            color,
            edge_id: (1, 2),
        };
        state.apply_action(action);

        // Should be able to build at node 2
        let actions = state.settlement_possibilities(color, false);
        assert_eq!(actions.len(), 1);
        assert!(actions.iter().any(|action| {
            matches!(action, Action::BuildSettlement { color: c, node_id } if *c == color && *node_id == 2)
        }));
    }

    #[test]
    fn test_initial_settlement_possibilities() {
        let state = State::new_base();
        let curr_color = state.get_current_color();
        let actions = state.settlement_possibilities(curr_color, true);
        assert_eq!(actions.len(), 54); // All nodes should be buildable

        for action in actions {
            match action {
                Action::BuildSettlement { color, .. } => assert_eq!(color, curr_color),
                _ => panic!("Expected BuildSettlement action"),
            }
        }
    }

    #[test]
    fn test_initial_road_possibilities() {
        let mut state = State::new_base();
        let curr_color = state.get_current_color();

        state.build_settlement(curr_color, 0);

        let actions = state.initial_road_possibilities(curr_color);
        assert_eq!(actions.len(), 3);

        for action in actions {
            match action {
                Action::BuildRoad { color, edge_id } => {
                    assert_eq!(color, curr_color);
                    assert!(edge_id.0 == 0 || edge_id.1 == 0);
                }
                _ => panic!("Expected BuildRoad action"),
            }
        }
    }

    #[test]
    fn test_settlement_blocks_neighbors() {
        let mut state = State::new_base();
        let curr_color = state.get_current_color();

        state.build_settlement(curr_color, 0);

        let actions = state.settlement_possibilities(curr_color, true);
        let neighbors = state.map_instance.get_neighbor_nodes(0);

        for action in actions {
            match action {
                Action::BuildSettlement { color: _, node_id } => {
                    assert_ne!(node_id, 0);
                    assert!(!neighbors.contains(&node_id));
                }
                _ => panic!("Expected BuildSettlement action"),
            }
        }
    }

    #[test]
    fn test_play_turn_initial_possibilities() {
        let mut state = State::new_base();

        assert!(state.is_initial_build_phase());
        assert!(matches!(
            state.get_action_prompt(),
            ActionPrompt::BuildInitialSettlement
        ));

        let actions = state.generate_playable_actions();
        match &actions[0] {
            Action::BuildSettlement { .. } => (),
            _ => panic!("Expected BuildSettlement action to be first action"),
        }
        state.apply_action(actions[0]);

        assert!(matches!(
            state.get_action_prompt(),
            ActionPrompt::BuildInitialRoad
        ));
    }

    #[test]
    fn test_robber_possibilities() {
        let mut state = State::new_base();
        let color1 = 1;
        let color2 = 2;

        state.build_settlement(color2, 0);

        // Give resources to color2
        let hand = state.get_mut_player_hand(color2);
        hand[1] = 1; // Use Brick's index (1) directly

        let actions = state.robber_possibilities(color1);

        // Should be able to move robber to any land tile except current location
        let num_land_tiles = state.map_instance.get_land_tiles().len();
        assert_eq!(actions.len(), num_land_tiles - 1);

        // Count how many actions involve stealing
        let steal_actions_count = actions.iter().filter(|action| {
            matches!(action, Action::MoveRobber { victim_opt: Some(v), .. } if *v == color2)
        }).count();

        // Node 0 is connected to 3 tiles, but one might be the robber's current location
        assert!(
            steal_actions_count == 2 || steal_actions_count == 3,
            "Should have 2 or 3 tiles where stealing is possible"
        );

        // Verify can't steal from player with no resources
        let hand = state.get_mut_player_hand(color2);
        hand[1] = 0; // Clear Brick resource
        let actions = state.robber_possibilities(color1);
        assert_eq!(actions.len(), num_land_tiles - 1); // All tiles except current

        // Verify no stealing actions when victim has no resources
        let steal_actions_count = actions
            .iter()
            .filter(|action| {
                matches!(
                    action,
                    Action::MoveRobber {
                        victim_opt: Some(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            steal_actions_count, 0,
            "Should have no stealing actions when victim has no resources"
        );
    }

    #[test]
    fn test_robber_cant_stay_in_place() {
        let state = State::new_base();
        let color = state.get_current_color();
        let actions = state.robber_possibilities(color);

        // Get current robber tile
        let current_robber_tile = state.get_robber_tile();

        // Verify no action tries to move robber to current location
        assert!(actions.iter().all(|action| {
            if let Action::MoveRobber { coordinate, .. } = action {
                state.map_instance.get_land_tile(*coordinate).unwrap().id != current_robber_tile
            } else {
                false
            }
        }));
    }

    #[test]
    fn test_year_of_plenty_possibilities() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give player a Year of Plenty card
        state.add_dev_card(color, DevCard::YearOfPlenty as usize);

        // Test with full bank - should have 15 actions:
        // - 5 same-resource actions (Wood+Wood, Brick+Brick, etc.)
        // - 10 different-resource combinations (5 choose 2)
        let actions = state.year_of_plenty_possibilities(color);
        assert_eq!(actions.len(), 15);

        // Test with empty bank - should have 0 actions
        for i in 0..5 {
            // 5 resource types
            state.set_bank_resource(i, 0);
        }
        let actions = state.year_of_plenty_possibilities(color);
        assert_eq!(actions.len(), 0);

        // Test with only 2 ORE available
        state.set_bank_resource(4, 2);
        let actions = state.year_of_plenty_possibilities(color);
        assert_eq!(actions.len(), 1); // Can only take ORE+ORE
        assert!(matches!(
            actions[0],
            Action::PlayYearOfPlenty {
                resources: (4, Some(4)),
                ..
            }
        ));

        // Test with only 1 WHEAT available
        for i in 0..5 {
            state.set_bank_resource(i, 0);
        }
        state.set_bank_resource(3, 1);
        let actions = state.year_of_plenty_possibilities(color);
        assert_eq!(actions.len(), 1); // Can take just one WHEAT when that's all that's available
        assert!(matches!(
            actions[0],
            Action::PlayYearOfPlenty {
                resources: (3, None),
                ..
            }
        ));
    }

    #[test]
    fn test_discard_possibilities() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give player 8 cards (above discard limit)
        {
            let hand = state.get_mut_player_hand(color);
            hand[0] = 8; // Give 8 wood cards
        }

        let actions = state.discard_possibilities(color);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Discard { color: c } if c == color));

        // Test with 7 cards (at discard limit)
        {
            let hand = state.get_mut_player_hand(color);
            hand[0] = 7;
        }
        let actions = state.discard_possibilities(color);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::EndTurn { color: c } if c == color));
    }

    #[test]
    fn test_maritime_trade_possibilities() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give player 4 wood (enough for 4:1 trade)
        {
            let hand = state.get_mut_player_hand(color);
            hand[0] = 4;
        }

        let actions = state.maritime_trade_possibilities(color);

        // Should be able to trade 4 wood for any other resource
        assert_eq!(actions.len(), 4); // Can trade for brick, sheep, wheat, ore
        assert!(actions.iter().all(|action| matches!(
            action,
            Action::MaritimeTrade {
                color: c,
                give,
                take,
                ratio
            } if *c == color && *give == 0 && *ratio == 4 && *take != 0
        )));

        // Test with insufficient resources
        {
            let hand = state.get_mut_player_hand(color);
            hand[0] = 3;
        }
        let actions = state.maritime_trade_possibilities(color);
        assert_eq!(actions.len(), 0);
    }

    #[test]
    fn test_maritime_trade_with_empty_bank() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give player resources for trading
        let hand = state.get_mut_player_hand(color);
        hand[0] = 4; // Give 4 wood for 4:1 trade

        // Empty the bank
        for i in 0..5 {
            state.set_bank_resource(i, 0);
        }

        let actions = state.maritime_trade_possibilities(color);
        assert_eq!(
            actions.len(),
            0,
            "Should not be able to trade with empty bank"
        );
    }

    #[test]
    fn test_maritime_trade_with_two_to_one_port() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Find a 2:1 wood port node
        let wood_port_node = find_port_node_by_type(&state, Some(Resource::Wood)).unwrap();

        // Build settlement at wood port
        state.build_settlement(color, wood_port_node);

        // Give player 2 wood (enough for 2:1 trade)
        let hand = state.get_mut_player_hand(color);
        hand[0] = 2;

        let actions = state.maritime_trade_possibilities(color);
        assert!(
            actions.iter().any(|action| matches!(
                action,
                Action::MaritimeTrade {
                    color: c,
                    give,
                    take,
                    ratio
                } if *c == color && *give == 0 && *ratio == 2 && *take != 0
            )),
            "Should be able to use 2:1 wood port"
        );
    }

    #[test]
    fn test_maritime_trade_with_three_to_one_port() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Find a 3:1 port node
        let port_node = find_port_node_by_type(&state, None).unwrap();

        // Build settlement at 3:1 port
        state.build_settlement(color, port_node);

        // Give player 3 brick (enough for 3:1 trade)
        let hand = state.get_mut_player_hand(color);
        hand[1] = 3; // Give 3 brick

        let actions = state.maritime_trade_possibilities(color);
        assert!(
            actions.iter().any(|action| matches!(
                action,
                Action::MaritimeTrade {
                    color: c,
                    give,
                    take,
                    ratio
                } if *c == color && *give == 1 && *ratio == 3 && *take != 0
            )),
            "Should be able to use 3:1 port for brick"
        );
    }
}
