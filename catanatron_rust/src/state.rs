use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    enums::{ActionPrompt, GameConfiguration, MapType},
    global_state::GlobalState,
    map_instance::{EdgeId, MapInstance, NodeId},
    state_vector::{
        actual_victory_points_index, initialize_state, player_hand_slice, seating_order_slice,
        StateVector, CURRENT_TICK_SEAT_INDEX, FREE_ROADS_AVAILABLE_INDEX, IS_DISCARDING_INDEX,
        IS_INITIAL_BUILD_PHASE_INDEX, IS_MOVING_ROBBER_INDEX,
    },
};

#[derive(Debug, PartialEq)]
enum Building {
    Settlement(u8), // Color
    City(u8),       // Color
}

#[derive(Debug)]
pub(crate) struct State {
    // These two are immutable
    config: Rc<GameConfiguration>,
    map_instance: Rc<MapInstance>,

    // This is mutable
    vector: StateVector,

    // These are caches for speeding up game state calculations
    board_buildable_ids: HashSet<NodeId>,
    buildings: HashMap<NodeId, Building>,
    roads: HashMap<EdgeId, u8>, // (Node1, Node2) -> Color
    roads_by_color: Vec<u8>,    // Color -> Count
    connected_components: HashMap<u8, Vec<HashSet<NodeId>>>,
    longest_road_color: Option<u8>,
    longest_road_length: u8,
}

mod move_generation;
mod mutations;

impl State {
    pub fn new(config: Rc<GameConfiguration>, map_instance: Rc<MapInstance>) -> Self {
        let vector = initialize_state(config.num_players);

        let board_buildable_ids = map_instance.land_nodes().clone();
        let buildings = HashMap::new();
        let roads = HashMap::new();
        let roads_by_color = vec![0; config.num_players as usize];
        let connected_components = HashMap::new();
        let longest_road_color = None;
        let longest_road_length = 0;

        Self {
            config,
            map_instance,
            vector,
            board_buildable_ids,
            buildings,
            roads,
            roads_by_color,
            connected_components,
            longest_road_color,
            longest_road_length,
        }
    }

    pub fn new_base() -> Self {
        let global_state = GlobalState::new();
        let config = GameConfiguration {
            dicard_limit: 7,
            vps_to_win: 10,
            map_type: MapType::Base,
            num_players: 4,
            max_ticks: 10,
        };
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );
        State::new(Rc::new(config), Rc::new(map_instance))
    }

    fn get_num_players(&self) -> u8 {
        self.config.num_players
    }

    fn get_num_players_usize(&self) -> usize {
        self.get_num_players() as usize
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

    fn is_road_building(&self) -> bool {
        self.vector[IS_INITIAL_BUILD_PHASE_INDEX] == 1
            && self.vector[FREE_ROADS_AVAILABLE_INDEX] == 1
    }

    pub fn get_current_tick_seat(&self) -> u8 {
        self.vector[CURRENT_TICK_SEAT_INDEX]
    }

    pub fn get_current_color(&self) -> u8 {
        let seating_order = self.get_seating_order();
        let current_tick_seat = self.get_current_tick_seat();
        seating_order[current_tick_seat as usize]
    }

    /// Returns a slice of Colors in the order of seating
    /// e.g. [2, 1, 0, 3] if Orange goes first, then Blue, then Red, and then White
    pub fn get_seating_order(&self) -> &[u8] {
        &self.vector[seating_order_slice(self.config.num_players as usize)]
    }

    pub fn get_action_prompt(&self) -> ActionPrompt {
        if self.is_initial_build_phase() {
            let num_things_built = self.buildings.len() + self.roads.len() / 2;
            if num_things_built == 2 * self.config.num_players as usize {
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

    pub fn get_mut_player_hand(&mut self, color: u8) -> &mut [u8] {
        &mut self.vector[player_hand_slice(color)]
    }

    pub fn get_player_hand(&self, color: u8) -> &[u8] {
        &self.vector[player_hand_slice(color)]
    }

    pub fn get_actual_victory_points(&self, color: u8) -> u8 {
        self.vector[actual_victory_points_index(self.config.num_players, color)]
    }

    pub fn winner(&self) -> Option<u8> {
        let current_color = self.get_current_color();

        let actual_victory_points = self.get_actual_victory_points(current_color);
        if actual_victory_points >= self.config.vps_to_win {
            return Some(current_color);
        }
        None
    }

    // ===== Board Getters =====
    // TODO: Potentially cache this implementation
    pub fn board_buildable_edges(&self, color: u8) -> Vec<EdgeId> {
        let color_components = self.connected_components.get(&color).unwrap();
        let expandable_nodes: Vec<NodeId> = color_components
            .iter()
            .flat_map(|component| component.iter())
            .cloned()
            .collect();

        let mut buildable = HashSet::new();
        for node in expandable_nodes {
            for edge in self.map_instance.get_neighbor_edges(node) {
                if !self.roads.contains_key(&edge) {
                    let sorted_edge = (edge.0.min(edge.1), edge.0.max(edge.1));
                    buildable.insert(sorted_edge);
                }
            }
        }
        buildable.into_iter().collect()
    }

    fn get_connected_component_index(&self, color: u8, a: u8) -> Option<usize> {
        let components = self.connected_components.get(&color).unwrap();
        for (i, component) in components.iter().enumerate() {
            if component.contains(&a) {
                return Some(i);
            }
        }
        None
    }

    fn is_enemy_node(&self, color: u8, a: u8) -> bool {
        let node_color = self.get_node_color(a);
        match node_color {
            None => false,
            Some(node_color) => node_color != color,
        }
    }

    fn get_node_color(&self, a: u8) -> Option<u8> {
        match self.buildings.get(&a) {
            Some(Building::Settlement(color)) => Some(*color),
            Some(Building::City(color)) => Some(*color),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = State::new_base();

        assert_eq!(state.longest_road_color, None);
    }

    #[test]
    fn test_initial_build_phase() {
        let state = State::new_base();

        assert!(state.is_initial_build_phase());
        assert!(!state.is_moving_robber());
        assert!(!state.is_discarding());
    }
}
