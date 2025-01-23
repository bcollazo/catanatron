use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::{
    deck_slices::*,
    enums::{Action, DevCard},
    map_instance::{EdgeId, NodeId},
    state::Building,
    state_vector::*,
};

use super::State;

impl State {
    pub fn apply_action(&mut self, action: Action) {
        match action {
            Action::BuildSettlement(color, node_id) => {
                let (new_owner, new_length) = self.build_settlement(color, node_id);
                self.maintain_longest_road(new_owner, new_length);
            }
            Action::BuildRoad(color, edge_id) => {
                let (new_owner, new_length) = self.build_road(color, edge_id);
                self.maintain_longest_road(new_owner, new_length);
            }
            Action::BuildCity(color, node_id) => {
                self.build_city(color, node_id);
            }
            Action::BuyDevelopmentCard(color) => {
                self.buy_development_card(color);
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
            Action::PlayKnight(color) => {
                self.play_knight(color);
                self.maintain_largest_army();
            }
            Action::PlayYearOfPlenty(color, resources) => {
                self.play_year_of_plenty(color, resources);
            }
            Action::PlayMonopoly(color, resource) => {
                self.play_monopoly(color, resource);
            }
            Action::PlayRoadBuilding(color) => {
                self.play_road_building(color);
            }
            Action::MaritimeTrade(color, (give, take, ratio)) => {
                self.maritime_trade(color, give, take, ratio);
            }
            Action::EndTurn(color) => {
                self.end_turn(color);
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

    pub fn sub_victory_points(&mut self, color: u8, points: u8) {
        let n = self.get_num_players();
        self.vector[actual_victory_points_index(n, color)] -= points;
    }

    pub fn advance_turn(&mut self, step_size: i8) {
        // We add an extra num_players to ensure next_index is positive (u8)
        let num_players = self.get_num_players() as i8;
        let next_index =
            ((self.get_current_tick_seat() as i8 + step_size + num_players) % num_players) as u8;

        self.vector[CURRENT_TURN_SEAT_INDEX] = next_index;
        self.vector[CURRENT_TICK_SEAT_INDEX] = next_index;
    }

    pub fn build_settlement(&mut self, placing_color: u8, node_id: u8) -> (Option<u8>, u8) {
        self.buildings
            .insert(node_id, Building::Settlement(placing_color, node_id));
        self.buildings_by_color
            .entry(placing_color)
            .or_default()
            .push(Building::Settlement(placing_color, node_id));

        let is_free = self.is_initial_build_phase();
        if !is_free {
            freqdeck_sub(self.get_mut_player_hand(placing_color), SETTLEMENT_COST);
            freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], SETTLEMENT_COST);
        }

        self.add_victory_points(placing_color, 1);

        let mut road_lengths: HashMap<u8, u8> = HashMap::new();

        if is_free {
            let owned_buildings = self.buildings_by_color.get(&placing_color).unwrap();
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

                    let hand = self.get_mut_player_hand(placing_color);
                    freqdeck_add(hand, total_resources);
                }
            }
            // Maintain caches and longest road =====
            //   - connected_components
            let component = HashSet::from([node_id]);
            self.connected_components
                .entry(placing_color)
                .or_default()
                .push(component);
        } else {
            // Mantain connected_components
            // Mantain longest_road_color and longest_road_length
            let mut plowed_edges_by_color: HashMap<u8, Vec<EdgeId>> = HashMap::new();
            for edge in self.map_instance.get_neighbor_edges(node_id) {
                if let Some(&road_color) = self.roads.get(&edge) {
                    plowed_edges_by_color
                        .entry(road_color)
                        .or_default()
                        .push(edge);
                }
            }

            for (plowed_color, plowed_edges) in plowed_edges_by_color {
                if plowed_edges.len() != 2 || plowed_color == placing_color {
                    continue; // Skip if no bisection/plow
                }

                if let Some(plowed_component_idx) =
                    self.get_connected_component_index(plowed_color, node_id)
                {
                    let outer_nodes: Vec<NodeId> = plowed_edges
                        .iter()
                        .map(|&edge| if edge.0 == node_id { edge.1 } else { edge.0 })
                        .collect();

                    // First remove the bisected component
                    let road_components = self.connected_components.get_mut(&plowed_color).unwrap();
                    road_components.remove(plowed_component_idx);

                    let mut new_components = Vec::new();
                    for outer_node in outer_nodes {
                        let new_component = self.dfs_walk(outer_node, plowed_color);
                        if !new_component.is_empty() {
                            new_components.push(new_component);
                        }
                    }

                    let road_components = self.connected_components.get_mut(&plowed_color).unwrap();
                    road_components.extend(new_components);
                }

                // Insert the longest road length for all colors if a road was plowed
                for (&color, components) in &self.connected_components {
                    let max_length = components
                        .iter()
                        .map(|component| self.longest_acyclic_path(component, color).len())
                        .max()
                        .unwrap_or(0);
                    road_lengths.insert(color, max_length as u8);
                }
            }
        }
        // - board_buildable_ids
        self.board_buildable_ids.remove(&node_id);
        for neighbor_id in self.map_instance.get_neighbor_nodes(node_id) {
            self.board_buildable_ids.remove(&neighbor_id);
        }

        // Determine new longest road holder
        let (new_road_color, new_road_length) = if road_lengths.is_empty() {
            // If no road lengths affected, just return the previous longest road
            (self.longest_road_color, self.longest_road_length)
        } else {
            let max_entry = road_lengths
                .iter()
                .filter(|(_, &len)| len >= 5)
                .max_by_key(|(_, &len)| len);

            match max_entry {
                Some((&color, &length)) => (Some(color), length),
                None => (None, 0), // No player has >= 5 roads
            }
        };
        (new_road_color, new_road_length)
    }

    fn build_road(&mut self, placing_color: u8, edge_id: EdgeId) -> (Option<u8>, u8) {
        let inverted_edge = (edge_id.1, edge_id.0);
        self.roads.insert(edge_id, placing_color);
        self.roads.insert(inverted_edge, placing_color);
        self.roads_by_color[placing_color as usize] += 1;

        let is_initial_build_phase = self.is_initial_build_phase();
        let is_free = is_initial_build_phase || self.is_road_building();
        if !is_free {
            freqdeck_sub(self.get_mut_player_hand(placing_color), ROAD_COST);
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
        let a_index = self.get_connected_component_index(placing_color, a);
        let b_index = self.get_connected_component_index(placing_color, b);

        let affected_component = if a_index.is_none() && !self.is_enemy_node(placing_color, a) {
            // There has to be a component from b (since roads can only be built in a connected fashion)
            let component = self
                .connected_components
                .get_mut(&placing_color)
                .unwrap()
                .get_mut(b_index.unwrap())
                .unwrap();
            component.insert(a); // extend said component by 1 more node
            component.clone()
        } else if b_index.is_none() && !self.is_enemy_node(placing_color, b) {
            // There has to be a component from a (since roads can only be built in a connected fashion)
            let component = self
                .connected_components
                .get_mut(&placing_color)
                .unwrap()
                .get_mut(a_index.unwrap())
                .unwrap();
            component.insert(b);
            component.clone()
        } else if a_index.is_some() && b_index.is_some() && a_index != b_index {
            // Merge components into one and delete the other
            let smaller_idx = a_index.unwrap().min(b_index.unwrap());
            let larger_idx = a_index.unwrap().max(b_index.unwrap());
            let removed_component = self
                .connected_components
                .get_mut(&placing_color)
                .unwrap()
                .remove(larger_idx);
            let kept_component = self
                .connected_components
                .get_mut(&placing_color)
                .unwrap()
                .get_mut(smaller_idx)
                .unwrap();
            kept_component.extend(&removed_component);
            kept_component.clone()
        } else {
            // Edge is within same component, just get that component
            // In this case, a_index == b_index, which means that the edge
            // is already part of one component. No actions needed.
            self.connected_components
                .get(&placing_color)
                .unwrap()
                .get(a_index.unwrap())
                .unwrap()
                .clone()
        };

        let prev_road_color = self.longest_road_color;

        // Calculate length for affected component
        let path_length = self
            .longest_acyclic_path(&affected_component, placing_color)
            .len() as u8;

        let (new_road_color, new_road_length) =
            if path_length >= 5 && path_length > self.longest_road_length {
                (Some(placing_color), path_length)
            } else {
                (prev_road_color, self.longest_road_length)
            };
        (new_road_color, new_road_length)
    }

    fn build_city(&mut self, color: u8, node_id: u8) {
        self.buildings
            .insert(node_id, Building::City(color, node_id));
        let buildings = self.buildings_by_color.entry(color).or_default();
        if let Some(pos) = buildings.iter().position(|b| {
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

    fn buy_development_card(&mut self, color: u8) -> Option<DevCard> {
        // Get next card from deck
        if let Some(card) = take_next_dev_card(&mut self.vector) {
            // Pay for the card
            freqdeck_sub(self.get_mut_player_hand(color), DEVCARD_COST);
            freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], DEVCARD_COST);

            let dev_card = match card {
                0 => DevCard::Knight,
                1 => DevCard::YearOfPlenty,
                2 => DevCard::Monopoly,
                3 => DevCard::RoadBuilding,
                4 => DevCard::VictoryPoint,
                _ => panic!("Invalid dev card index"),
            };

            match dev_card {
                DevCard::VictoryPoint => {
                    self.add_victory_points(color, 1);
                }
                _ => {
                    let dev_hand =
                        &mut self.vector[player_devhand_slice(self.config.num_players, color)];
                    dev_hand[card as usize] += 1;
                }
            }

            Some(dev_card)
        } else {
            None
        }
    }

    fn roll_dice(&mut self, color: u8, dice_opt: Option<(u8, u8)>) {
        self.vector[HAS_ROLLED_INDEX] = 1;
        let (die1, die2) = dice_opt.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            (rng.gen_range(1..=6), rng.gen_range(1..=6))
        });
        let total = die1 + die2;

        if total == 7 {
            self.handle_roll_seven(color);
        } else {
            self.distribute_roll_yields(total);
            self.vector[CURRENT_TICK_SEAT_INDEX] = color;
        }
    }

    fn handle_roll_seven(&mut self, color: u8) {
        // Check who needs to discard
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
    }

    // Returns Vec of (color, resource_index, amount) tuples for what each player should receive
    fn collect_roll_yields(&self, roll: u8) -> Vec<(u8, usize, u8)> {
        let mut all_yields = Vec::new();
        let matching_tiles = self.map_instance.get_tiles_by_number(roll);

        for tile in matching_tiles {
            // Skip robber tile
            if self.vector[ROBBER_TILE_INDEX] == tile.id {
                continue;
            }

            if let Some(resource) = tile.resource {
                let resource_idx = resource as usize;
                // Collect all yields for this tile
                for &node_id in tile.hexagon.nodes.values() {
                    if let Some(building) = self.buildings.get(&node_id) {
                        match building {
                            Building::Settlement(owner_color, _) => {
                                all_yields.push((*owner_color, resource_idx, 1));
                            }
                            Building::City(owner_color, _) => {
                                all_yields.push((*owner_color, resource_idx, 2));
                            }
                        }
                    }
                }
            }
        }
        all_yields
    }

    fn distribute_roll_yields(&mut self, roll: u8) {
        let yields = self.collect_roll_yields(roll);
        if yields.is_empty() {
            return;
        }

        // Calculate total needed by resource type
        let mut resource_needs = [0u8; 5];
        for (_, resource_idx, amount) in &yields {
            resource_needs[*resource_idx] += amount;
        }

        // Check what can be allocated from bank
        let bank = &self.vector[BANK_RESOURCE_SLICE];
        let mut distributable = resource_needs;
        let mut insufficient = false;
        let multiple_recipients = yields
            .iter()
            .map(|(color, _, _)| color)
            .collect::<HashSet<_>>()
            .len()
            > 1;

        for i in 0..5 {
            if bank[i] < resource_needs[i] {
                if multiple_recipients {
                    // If not enough for everyone, no one gets anything
                    return;
                }
                distributable[i] = bank[i];
                insufficient = true;
            }
        }

        // If we got here, we can distribute something
        if insufficient {
            // Single player case - give what we can
            for (owner_color, resource_idx, amount) in yields {
                let available = distributable[resource_idx].min(amount);
                if available > 0 {
                    self.vector[BANK_RESOURCE_SLICE][resource_idx] -= available;
                    self.get_mut_player_hand(owner_color)[resource_idx] += available;
                }
            }
        } else {
            // Full distribution case
            for (owner_color, resource_idx, amount) in yields {
                self.vector[BANK_RESOURCE_SLICE][resource_idx] -= amount;
                self.get_mut_player_hand(owner_color)[resource_idx] += amount;
            }
        }
    }

    /*
     * TODO: For now, we're not letting players choose what to discard, to avoid
     * the combinatorial explosion of possibilities. Instead, we'll just
     * force discards in a way that maximizes resource diversity.
     */
    fn discard(&mut self, color: u8) {
        let mut remaining_hand = self.get_player_hand(color).to_vec();
        let total_cards: u8 = remaining_hand.iter().sum();
        let mut to_discard = total_cards - (total_cards / 2);
        let mut discarded = [0u8; 5];

        while to_discard > 0 {
            // Find highest frequency resources
            let max_count = *remaining_hand.iter().max().unwrap();
            let max_indices: Vec<_> = (0..5).filter(|&i| remaining_hand[i] == max_count).collect();

            // Take one card from each highest frequency resource
            for &i in &max_indices {
                if to_discard > 0 {
                    remaining_hand[i] -= 1;
                    discarded[i] += 1;
                    to_discard -= 1;
                }
            }
        }

        freqdeck_sub(self.get_mut_player_hand(color), discarded);
        freqdeck_add(&mut self.vector[BANK_RESOURCE_SLICE], discarded);
        self.vector[IS_DISCARDING_INDEX] = 0;
        // TODO: Advance turn; handle discarders left and pass turn to original roller
    }

    fn move_robber(&mut self, color: u8, coordinate: (i8, i8, i8), victim_opt: Option<u8>) {
        self.vector[ROBBER_TILE_INDEX] = self.map_instance.get_land_tile(coordinate).unwrap().id;

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

    fn maintain_longest_road(&mut self, new_owner: Option<u8>, new_length: u8) {
        let prev_owner = self.longest_road_color;
        self.longest_road_color = new_owner;
        self.longest_road_length = new_length;

        if new_owner == prev_owner {
            return;
        }

        if let Some(prev_owner) = prev_owner {
            self.sub_victory_points(prev_owner, 2);
        }

        if let Some(new_owner) = new_owner {
            self.add_victory_points(new_owner, 2);
        }
    }

    fn dfs_walk(&self, start_node: NodeId, color: u8) -> HashSet<NodeId> {
        let mut agenda = vec![start_node];
        let mut visited = HashSet::new();

        while let Some(node) = agenda.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node);

            if self.is_enemy_node(color, node) {
                continue;
            }

            for neighbor in self.map_instance.get_neighbor_nodes(node) {
                let edge = (node.min(neighbor), node.max(neighbor));
                if self.roads.get(&edge) == Some(&color) {
                    agenda.push(neighbor);
                }
            }
        }
        visited
    }

    fn play_knight(&mut self, color: u8) {
        // Mark card as played
        self.remove_dev_card(color, DevCard::Knight as usize);
        self.add_played_dev_card(color, DevCard::Knight as usize);
        self.set_has_played_dev_card();

        // Set state to move robber
        self.set_is_moving_robber();
    }

    fn maintain_largest_army(&mut self) {
        let prev_owner = self.largest_army_color;
        let prev_count = self.largest_army_count;

        // Find player with most knights (if any have 3 or more)
        let mut max_knights = 0;
        let mut max_knights_color = None;

        for color in 0..self.get_num_players() {
            let knights = self.get_played_dev_card_count(color, DevCard::Knight as usize);
            if knights >= 3 && knights > max_knights {
                max_knights = knights;
                max_knights_color = Some(color);
            }
        }

        // Case where playerB meets playerA's largest army -> no change
        if max_knights == prev_count {
            return;
        }

        self.largest_army_color = max_knights_color;
        self.largest_army_count = max_knights;

        // If playerA retains largest army -> no VP changes
        if max_knights_color == prev_owner {
            return;
        }

        if let Some(prev_owner) = prev_owner {
            self.sub_victory_points(prev_owner, 2);
        }

        if let Some(new_owner) = max_knights_color {
            self.add_victory_points(new_owner, 2);
        }
    }

    fn play_year_of_plenty(&mut self, color: u8, resources: [u8; 2]) {
        // Assume move_generation has already checked that player has year of plenty card
        // and that bank has enough resources
        self.remove_dev_card(color, DevCard::YearOfPlenty as usize);
        self.add_played_dev_card(color, DevCard::YearOfPlenty as usize);
        self.set_has_played_dev_card();

        // Give resources to player
        for resource in resources {
            self.take_from_bank_give_to_player(color, resource);
        }
    }

    fn play_monopoly(&mut self, color: u8, resource: u8) {
        // Assume move_generation has already checked that player has monopoly card.
        self.remove_dev_card(color, DevCard::Monopoly as usize);
        self.add_played_dev_card(color, DevCard::Monopoly as usize);
        self.set_has_played_dev_card();

        // Steal all resources of type from other players
        for victim_color in 0..self.get_num_players() {
            if victim_color != color {
                let amount = self.get_player_resource_count(victim_color, resource);
                if amount > 0 {
                    self.take_from_player_give_to_player(victim_color, color, resource, amount);
                }
            }
        }
    }

    fn play_road_building(&mut self, color: u8) {
        // Assume move_generation has already checked that player has road building card.
        self.remove_dev_card(color, DevCard::RoadBuilding as usize);
        self.add_played_dev_card(color, DevCard::RoadBuilding as usize);
        self.set_has_played_dev_card();

        // Set state for free roads
        self.vector[IS_BUILDING_ROAD_INDEX] = 1;
        self.vector[FREE_ROADS_AVAILABLE_INDEX] = 2;
    }

    fn maritime_trade(&mut self, color: u8, give: u8, take: u8, ratio: u8) {
        // Assume move_generation has already checked that player has enough resources
        // to give and that bank has enough resources to take
        self.take_from_player_give_to_bank(color, give, ratio);
        self.take_from_bank_give_to_player(color, take);
    }

    fn end_turn(&mut self, _color: u8) {
        self.vector[HAS_PLAYED_DEV_CARD] = 0;
        self.vector[HAS_ROLLED_INDEX] = 0;

        self.advance_turn(1);
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

    #[test]
    fn test_settlement_cuts_longest_road() {
        let mut state = State::new_base();
        let color1 = 1;
        let color2 = 2;

        // give color1 6 consecutive roads
        state.apply_action(Action::BuildSettlement(color1, 0));
        for edge in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 16)] {
            state.apply_action(Action::BuildRoad(color1, edge));
        }

        assert_eq!(state.longest_road_color, Some(color1));
        assert_eq!(state.get_actual_victory_points(color1), 3);
        assert_eq!(state.get_actual_victory_points(color2), 0);

        // Give color2 a settlement at node 4 to bisect color1's Longest Road
        state.vector[IS_INITIAL_BUILD_PHASE_INDEX] = 0;
        freqdeck_add(state.get_mut_player_hand(color2), SETTLEMENT_COST);
        state.apply_action(Action::BuildSettlement(color2, 4));

        assert_eq!(state.longest_road_color, None);
        assert_eq!(state.get_actual_victory_points(color1), 1);
        assert_eq!(state.get_actual_victory_points(color2), 1);
    }

    #[test]
    fn test_build_road_maintains_connected_components() {
        let mut state = State::new_base();
        let color1 = 1;

        state.build_settlement(color1, 0);
        state.build_road(color1, (0, 1));

        let components = state.connected_components.get(&color1).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], HashSet::from([0, 1]));

        state.build_road(color1, (1, 2));

        let components = state.connected_components.get(&color1).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], HashSet::from([0, 1, 2]));

        state.build_settlement(color1, 4);
        state.build_road(color1, (3, 4));

        let components = state.connected_components.get(&color1).unwrap();
        assert_eq!(components.len(), 2);
        assert_eq!(components[0], HashSet::from([0, 1, 2]));
        assert_eq!(components[1], HashSet::from([3, 4]));

        state.build_road(color1, (2, 3));

        let components = state.connected_components.get(&color1).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], HashSet::from([0, 1, 2, 3, 4]));
    }

    #[test]
    fn test_settlement_cuts_longest_road_and_transfers() {
        let mut state = State::new_base();
        let color1 = 1;
        let color2 = 2;

        // give color1 6 consecutive roads
        state.apply_action(Action::BuildSettlement(color1, 0));
        for edge in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 16)] {
            state.apply_action(Action::BuildRoad(color1, edge));
        }
        // Give color2 5 consecutive roads with potential to bisect/plow color1's road
        state.apply_action(Action::BuildSettlement(color2, 11));
        for edge in [(11, 12), (12, 13), (13, 14), (14, 15), (4, 15)] {
            state.apply_action(Action::BuildRoad(color2, edge));
        }

        assert_eq!(state.longest_road_color, Some(color1));
        assert_eq!(state.get_actual_victory_points(color1), 3);
        assert_eq!(state.get_actual_victory_points(color2), 1);

        // Give color2 a settlement at node 4 to bisect color1's Longest Road
        state.vector[IS_INITIAL_BUILD_PHASE_INDEX] = 0;
        freqdeck_add(state.get_mut_player_hand(color2), SETTLEMENT_COST);
        state.apply_action(Action::BuildSettlement(color2, 4));

        assert_eq!(state.longest_road_color, Some(color2));
        assert_eq!(state.get_actual_victory_points(color1), 1);
        assert_eq!(state.get_actual_victory_points(color2), 4);
    }

    #[test]
    fn test_extend_own_longest_road() {
        let mut state = State::new_base();
        let color1 = 1;

        state.apply_action(Action::BuildSettlement(color1, 0));
        for edge in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] {
            state.apply_action(Action::BuildRoad(color1, edge));
        }

        assert_eq!(state.longest_road_color, Some(color1));
        assert_eq!(state.longest_road_length, 5);
        assert_eq!(state.get_actual_victory_points(color1), 3);

        state.apply_action(Action::BuildRoad(color1, (5, 16)));

        assert_eq!(state.longest_road_color, Some(color1));
        assert_eq!(state.longest_road_length, 6);
        assert_eq!(state.get_actual_victory_points(color1), 3);
    }

    #[test]
    fn test_bisection_counts_remaining_components() {
        let mut state = State::new_base();
        let color1 = 1;
        let color2 = 2;

        state.apply_action(Action::BuildSettlement(color1, 0));
        for edge in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 16)] {
            state.apply_action(Action::BuildRoad(color1, edge));
        }

        assert_eq!(state.longest_road_color, Some(color1));
        assert_eq!(state.longest_road_length, 6);
        assert_eq!(state.get_actual_victory_points(color1), 3);

        state.vector[IS_INITIAL_BUILD_PHASE_INDEX] = 0;
        freqdeck_add(state.get_mut_player_hand(color2), SETTLEMENT_COST);
        state.apply_action(Action::BuildSettlement(color2, 5));

        assert_eq!(state.longest_road_color, Some(color1));
        assert_eq!(state.longest_road_length, 5);
        assert_eq!(state.connected_components.get(&color1).unwrap().len(), 2);
        assert_eq!(state.get_actual_victory_points(color1), 3);
        assert_eq!(state.get_actual_victory_points(color2), 1);
    }

    #[test]
    fn test_buy_development_cards() {
        let mut state = State::new_base();
        let color = state.get_current_color();
        let mut cards_drawn = 0;

        while cards_drawn < 26 {
            freqdeck_add(state.get_mut_player_hand(color), DEVCARD_COST);
            let initial_hand: [u8; 5] = state.get_player_hand(color).try_into().unwrap();
            let initial_devhand = state.get_player_devhand(color).to_vec();
            let initial_bank = state.vector[BANK_RESOURCE_SLICE].to_vec();
            let initial_vps = state.get_actual_victory_points(color);

            let drawn_card = state.buy_development_card(color);
            cards_drawn += 1;

            println!("Cards Drawn: {}, Drawn card: {:?}", cards_drawn, drawn_card);

            if cards_drawn < 26 {
                let hand_after = state.get_player_hand(color);
                let bank_after = &state.vector[BANK_RESOURCE_SLICE];
                for i in 0..5 {
                    assert_eq!(hand_after[i], initial_hand[i] - DEVCARD_COST[i]);
                    assert_eq!(bank_after[i], initial_bank[i] + DEVCARD_COST[i]);
                }
                let devhand_after = state.get_player_devhand(color);

                if drawn_card == Some(DevCard::VictoryPoint) {
                    // VP added, devhand not incremented
                    assert_eq!(state.get_actual_victory_points(color), initial_vps + 1);
                    assert_eq!(
                        devhand_after[drawn_card.unwrap() as usize],
                        initial_devhand[drawn_card.unwrap() as usize]
                    );
                } else {
                    // VP not added, devhand incremented
                    assert_eq!(state.get_actual_victory_points(color), initial_vps);
                    assert_eq!(
                        devhand_after[drawn_card.unwrap() as usize],
                        initial_devhand[drawn_card.unwrap() as usize] + 1
                    );
                }
            } else {
                // 26th card should not be drawn
                assert!(drawn_card.is_none());
                assert_eq!(state.get_player_hand(color), initial_hand);
                assert_eq!(&state.vector[BANK_RESOURCE_SLICE], initial_bank);
            }
        }
    }

    #[test]
    fn test_roll_yields_resources() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        state.build_settlement(color, 0);

        let adjacent_tiles = state.map_instance.get_adjacent_tiles(0).unwrap();

        let mut chosen_roll = None;
        let mut expected_resource_yields = [0; 5];

        for tile in adjacent_tiles.iter() {
            if let (Some(number), Some(resource)) = (tile.number, tile.resource) {
                // First valid number we find will be our roll
                // Don't pick robber tile
                if tile.id != state.vector[ROBBER_TILE_INDEX] {
                    if chosen_roll.is_none() {
                        chosen_roll = Some(number);
                    }

                    if Some(number) == chosen_roll {
                        expected_resource_yields[resource as usize] += 1;
                    }
                }
            }
        }

        let initial_bank = state.vector[BANK_RESOURCE_SLICE].to_vec();
        let initial_hand = state.get_player_hand(color).to_vec();
        // Roll numbers should sum to chosen_roll
        let roll_numbers = (chosen_roll.unwrap() / 2, (chosen_roll.unwrap() + 1) / 2);

        state.apply_action(Action::Roll(color, Some(roll_numbers)));

        for resource_idx in 0..5 {
            assert_eq!(
                state.vector[BANK_RESOURCE_SLICE][resource_idx],
                initial_bank[resource_idx] - expected_resource_yields[resource_idx],
                "Bank should have {} fewer resource of {:?}",
                expected_resource_yields[resource_idx],
                resource_idx
            );
            assert_eq!(
                state.get_player_hand(color)[resource_idx],
                initial_hand[resource_idx] + expected_resource_yields[resource_idx],
                "Player should have {} more resource of {:?}",
                expected_resource_yields[resource_idx],
                resource_idx
            )
        }
    }

    #[test]
    fn test_roll_city_yields_double() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        freqdeck_add(state.get_mut_player_hand(color), CITY_COST);
        state.build_settlement(color, 0);
        state.build_city(color, 0);

        let adjacent_tiles = state.map_instance.get_adjacent_tiles(0).unwrap();

        let mut chosen_roll = None;
        let mut expected_resource_yields = [0; 5];

        for tile in adjacent_tiles.iter() {
            if let (Some(number), Some(resource)) = (tile.number, tile.resource) {
                // Don't pick robber tile
                if tile.id != state.vector[ROBBER_TILE_INDEX] {
                    if chosen_roll.is_none() {
                        chosen_roll = Some(number);
                    }

                    if Some(number) == chosen_roll {
                        expected_resource_yields[resource as usize] += 2;
                    }
                }
            }
        }

        let initial_bank = state.vector[BANK_RESOURCE_SLICE].to_vec();
        let initial_hand = state.get_player_hand(color).to_vec();
        // Roll numbers should sum to chosen_roll
        let roll_numbers = (chosen_roll.unwrap() / 2, (chosen_roll.unwrap() + 1) / 2);

        state.apply_action(Action::Roll(color, Some(roll_numbers)));

        for resource_idx in 0..5 {
            assert_eq!(
                state.vector[BANK_RESOURCE_SLICE][resource_idx],
                initial_bank[resource_idx] - expected_resource_yields[resource_idx],
                "Bank should have {} fewer resource of {:?}",
                expected_resource_yields[resource_idx],
                resource_idx
            );
            assert_eq!(
                state.get_player_hand(color)[resource_idx],
                initial_hand[resource_idx] + expected_resource_yields[resource_idx],
                "Player should have {} more resource of {:?}",
                expected_resource_yields[resource_idx],
                resource_idx
            );
        }
    }

    #[test]
    fn test_roll_single_player_partial_payment_when_insufficient_bank() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        let node_id = 0;
        state.build_settlement(color, node_id);
        freqdeck_add(state.get_mut_player_hand(color), CITY_COST);
        state.build_city(color, node_id);

        let adjacent_tiles = state.map_instance.get_adjacent_tiles(node_id).unwrap();

        let mut chosen_roll = None;
        let mut chosen_resource = None;

        for tile in adjacent_tiles.iter() {
            if let (Some(number), Some(resource)) = (tile.number, tile.resource) {
                if tile.id != state.vector[ROBBER_TILE_INDEX] && chosen_roll.is_none() {
                    chosen_roll = Some(number);
                    chosen_resource = Some(resource);
                }
            }
        }
        assert!(chosen_roll.is_some(), "Should find at least one valid tile");

        for i in 0..5 {
            state.vector[BANK_RESOURCE_SLICE][i] = 1;
        }
        let hand_before = state.get_player_hand(color).to_vec();

        let roll = chosen_roll.unwrap();
        let roll_numbers = (roll / 2, (roll + 1) / 2);
        state.roll_dice(color, Some(roll_numbers));

        let chosen_resource_idx = chosen_resource.unwrap() as usize;
        assert_eq!(state.vector[BANK_RESOURCE_SLICE][chosen_resource_idx], 0);

        assert_eq!(
            state.get_player_hand(color)[chosen_resource_idx],
            hand_before[chosen_resource_idx] + 1
        );
        assert_eq!(state.vector[BANK_RESOURCE_SLICE][chosen_resource_idx], 0)
    }

    #[test]
    fn test_roll_multiple_player_no_payment_when_insufficient_bank() {
        let mut state = State::new_base();
        let color1 = 1;
        let color2 = 2;

        let (resource, number, node1, node2) = {
            let tile = state
                .map_instance
                .get_land_tiles()
                .values()
                .find(|tile| {
                    tile.resource.is_some() && // Not a desert
                    tile.id != state.vector[ROBBER_TILE_INDEX] // Not under robber
                })
                .expect("Should be at least one valid tile");

            let node_ids: Vec<_> = tile.hexagon.nodes.values().take(2).copied().collect();

            (
                tile.resource.unwrap(),
                tile.number.unwrap(),
                node_ids[0],
                node_ids[1],
            )
        };

        // Place two opposing cities on a shared tile with expected yields
        state.build_settlement(color1, node1);
        state.build_settlement(color2, node2);
        freqdeck_add(state.get_mut_player_hand(color1), CITY_COST);
        freqdeck_add(state.get_mut_player_hand(color2), CITY_COST);
        state.build_city(color1, node1);
        state.build_city(color2, node2);

        // Set bank to have only 1 of the needed resource
        let resource_idx = resource as usize;
        state.vector[BANK_RESOURCE_SLICE][resource_idx] = 1;

        let bank_before = state.vector[BANK_RESOURCE_SLICE][resource_idx];
        let hand1_before = state.get_player_hand(color1)[resource_idx];
        let hand2_before = state.get_player_hand(color2)[resource_idx];

        // Roll the shared tile's number
        let roll_numbers = (number / 2, (number + 1) / 2);
        state.roll_dice(color1, Some(roll_numbers));

        assert_eq!(
            state.vector[BANK_RESOURCE_SLICE][resource_idx], bank_before,
            "Bank should be unchanged"
        );
        // Neither player should get any resources
        assert_eq!(
            state.get_player_hand(color1)[resource_idx],
            hand1_before,
            "Player 1 should not receive resources"
        );
        assert_eq!(
            state.get_player_hand(color2)[resource_idx],
            hand2_before,
            "Player 2 should not receive resources"
        );
    }

    #[test]
    fn test_discard() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give the player a known distribution of 17 cards
        freqdeck_add(state.get_mut_player_hand(color), [3, 9, 1, 3, 1]);

        let bank_before = state.vector[BANK_RESOURCE_SLICE].to_vec();

        state.discard(color);

        // After discarding, the player should have half => 17 / 2 = 8.
        let total_after: u8 = state.get_player_hand(color).iter().sum();
        assert_eq!(total_after, 8, "Player should have exactly 8 cards left.");

        // Verify discard phase ended
        assert_eq!(
            state.vector[IS_DISCARDING_INDEX], 0,
            "Discard phase should end."
        );

        // The bank should have received exactly 6 more cards in total
        let bank_after = &state.vector[BANK_RESOURCE_SLICE];
        let mut total_discarded = 0;
        for i in 0..5 {
            total_discarded += bank_after[i] - bank_before[i];
        }
        assert_eq!(
            total_discarded, 9,
            "Exactly 9 cards should have been added to the bank."
        );

        // Check the specific distribution after discard
        let final_player_hand = state.get_player_hand(color);
        assert_eq!(
            final_player_hand,
            &[2, 2, 1, 2, 1],
            "Discard logic should spread discards across highest-frequency resources first."
        );
    }

    #[test]
    fn test_play_knight() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        state.add_dev_card(color, DevCard::Knight as usize);
        assert_eq!(state.get_dev_card_count(color, DevCard::Knight as usize), 1);
        assert_eq!(
            state.get_played_dev_card_count(color, DevCard::Knight as usize),
            0
        );
        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 0);
        assert_eq!(state.vector[IS_MOVING_ROBBER_INDEX], 0);

        state.play_knight(color);

        assert_eq!(state.get_dev_card_count(color, DevCard::Knight as usize), 0);
        assert_eq!(
            state.get_played_dev_card_count(color, DevCard::Knight as usize),
            1
        );
        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 1);
        assert_eq!(state.vector[IS_MOVING_ROBBER_INDEX], 1);
    }

    #[test]
    fn test_play_knight_largest_army() {
        let mut state = State::new_base();
        let color1 = 1;
        let color2 = 2;

        // Give first player 3 knight cards
        for _ in 0..3 {
            state.add_dev_card(color1, DevCard::Knight as usize);
        }

        // Play knights and verify largest army
        for i in 0..3 {
            state.vector[HAS_PLAYED_DEV_CARD] = 0; // Reset for each turn
            state.apply_action(Action::PlayKnight(color1));

            // Verify knight was removed and marked as played
            assert_eq!(
                state.get_dev_card_count(color1, DevCard::Knight as usize),
                2 - i
            );
            assert_eq!(
                state.get_played_dev_card_count(color1, DevCard::Knight as usize),
                i + 1
            );
            assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 1);
            assert_eq!(state.vector[IS_MOVING_ROBBER_INDEX], 1);

            // Check largest army status
            if i == 2 {
                assert_eq!(state.largest_army_color, Some(color1));
                assert_eq!(state.largest_army_count, 3);
                assert_eq!(state.get_actual_victory_points(color1), 2);
                assert_eq!(state.get_actual_victory_points(color2), 0);
            } else {
                assert_eq!(state.largest_army_color, None);
                assert_eq!(state.largest_army_count, 0);
                assert_eq!(state.get_actual_victory_points(color1), 0);
                assert_eq!(state.get_actual_victory_points(color2), 0);
            }
        }

        // Now give second player 4 knight cards and have them take largest army
        for _ in 0..4 {
            state.add_dev_card(color2, DevCard::Knight as usize);
        }

        // Play knights with second player
        for i in 0..4 {
            state.vector[HAS_PLAYED_DEV_CARD] = 0; // Reset for each turn
            state.apply_action(Action::PlayKnight(color2));

            // Verify knight was removed and marked as played
            assert_eq!(
                state.get_dev_card_count(color2, DevCard::Knight as usize),
                3 - i
            );
            assert_eq!(
                state.get_played_dev_card_count(color2, DevCard::Knight as usize),
                i + 1
            );

            // Check largest army status
            if i == 3 {
                // After 4th knight, should take largest army
                assert_eq!(state.largest_army_color, Some(color2));
                assert_eq!(state.largest_army_count, 4);
                assert_eq!(state.get_actual_victory_points(color1), 0); // Lost 2 VPs
                assert_eq!(state.get_actual_victory_points(color2), 2); // Gained 2 VPs
            } else {
                // Still held by first player
                assert_eq!(state.largest_army_color, Some(color1));
                assert_eq!(state.largest_army_count, 3);
                assert_eq!(state.get_actual_victory_points(color1), 2);
                assert_eq!(state.get_actual_victory_points(color2), 0);
            }
        }
    }

    #[test]
    fn test_play_year_of_plenty() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give player a year of plenty card
        state.add_dev_card(color, DevCard::YearOfPlenty as usize);

        let bank_before = state.vector[BANK_RESOURCE_SLICE].to_vec();
        let hand_before = state.get_player_hand(color).to_vec();

        // Play year of plenty for wood and brick
        state.play_year_of_plenty(color, [0, 1]);

        // Verify card was removed from hand
        assert_eq!(
            state.get_dev_card_count(color, DevCard::YearOfPlenty as usize),
            0
        );

        // Verify card was marked as played
        assert_eq!(
            state.get_played_dev_card_count(color, DevCard::YearOfPlenty as usize),
            1
        );
        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 1);

        // Verify resources were transferred
        assert_eq!(state.vector[BANK_RESOURCE_SLICE][0], bank_before[0] - 1);
        assert_eq!(state.vector[BANK_RESOURCE_SLICE][1], bank_before[1] - 1);
        assert_eq!(state.get_player_hand(color)[0], hand_before[0] + 1);
        assert_eq!(state.get_player_hand(color)[1], hand_before[1] + 1);
    }

    #[test]
    fn test_play_monopoly() {
        let mut state = State::new_base();
        let monopolist_color = state.get_current_color();

        // Give player a monopoly card
        state.add_dev_card(monopolist_color, DevCard::Monopoly as usize);

        // Give other players some wood
        for other_color in 0..state.get_num_players() {
            if other_color != monopolist_color {
                state.get_mut_player_hand(other_color)[0] = 3;
            }
        }

        let initial_wood = state.get_player_hand(monopolist_color)[0];
        let expected_stolen = 3 * (state.get_num_players() - 1) as u8; // 3 wood from each other player

        // Play monopoly on wood (resource index 0)
        state.play_monopoly(monopolist_color, 0);

        assert_eq!(
            state.get_dev_card_count(monopolist_color, DevCard::Monopoly as usize),
            0
        );
        assert_eq!(
            state.get_played_dev_card_count(monopolist_color, DevCard::Monopoly as usize),
            1
        );
        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 1);
        assert_eq!(
            state.get_player_hand(monopolist_color)[0],
            initial_wood + expected_stolen
        );

        // Verify other players lost their wood
        for other_color in 0..state.get_num_players() {
            if other_color != monopolist_color {
                assert_eq!(state.get_player_hand(other_color)[0], 0);
            }
        }
    }

    #[test]
    fn test_play_road_building() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        // Give player a road building card
        state.add_dev_card(color, DevCard::RoadBuilding as usize);
        assert_eq!(
            state.get_dev_card_count(color, DevCard::RoadBuilding as usize),
            1
        );
        assert_eq!(
            state.get_played_dev_card_count(color, DevCard::RoadBuilding as usize),
            0
        );
        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 0);

        // Play road building card
        state.play_road_building(color);

        // Verify card was removed from hand
        assert_eq!(
            state.get_dev_card_count(color, DevCard::RoadBuilding as usize),
            0
        );

        // Verify card was marked as played
        assert_eq!(
            state.get_played_dev_card_count(color, DevCard::RoadBuilding as usize),
            1
        );
        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 1);

        // Verify state was set for free roads
        assert_eq!(state.vector[IS_BUILDING_ROAD_INDEX], 1);
        assert_eq!(state.vector[FREE_ROADS_AVAILABLE_INDEX], 2);
    }

    #[test]
    fn test_maritime_trade_basic_rate() {
        let mut state = State::new_base();
        let color = state.get_current_color();

        state.get_mut_player_hand(color)[0] = 4; // 4 wood

        let initial_bank_brick = state.vector[BANK_RESOURCE_SLICE][1];

        state.apply_action(Action::MaritimeTrade(color, (0, 1, 4)));

        assert_eq!(state.get_player_hand(color)[0], 0);
        assert_eq!(state.get_player_hand(color)[1], 1);
        assert_eq!(state.vector[BANK_RESOURCE_SLICE][0], 19 + 4);
        assert_eq!(state.vector[BANK_RESOURCE_SLICE][1], initial_bank_brick - 1);
    }

    #[test]
    fn test_end_turn() {
        let mut state = State::new_base();
        let starting_color = state.get_current_color();
        let seating_order = state.get_seating_order().to_vec();

        state.vector[HAS_PLAYED_DEV_CARD] = 1;
        state.vector[HAS_ROLLED_INDEX] = 1;
        state.apply_action(Action::EndTurn(starting_color));

        assert_eq!(state.vector[HAS_PLAYED_DEV_CARD], 0);
        assert_eq!(state.vector[HAS_ROLLED_INDEX], 0);

        assert_eq!(state.get_current_color(), seating_order[1]);

        for _ in 0..(state.get_num_players() - 1) {
            state.apply_action(Action::EndTurn(state.get_current_color()));
        }

        assert_eq!(state.get_current_color(), starting_color);
    }
}
