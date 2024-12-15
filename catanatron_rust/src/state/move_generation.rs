use crate::deck_slices::{freqdeck_contains, CITY_COST, ROAD_COST};
use crate::enums::{Action, ActionPrompt, DevCard};
use crate::state::State;

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
            ActionPrompt::PlayTurn => self.play_turn_possibilities(current_color),
            ActionPrompt::Discard => todo!("generate_playbale_actions for Discard"),
            ActionPrompt::MoveRobber => todo!("generate_playbale_actions for Move robber"),
            ActionPrompt::DecideTrade => todo!("generate_playbale_actions for Decide trade"),
            ActionPrompt::DecideAcceptees => {
                todo!("generate_playbale_actions for Decide acceptees")
            }
        }
    }

    pub fn settlement_possibilities(&self, color: u8, is_initial_build_phase: bool) -> Vec<Action> {
        if is_initial_build_phase {
            self.board_buildable_ids
                .iter()
                .map(|node_id| Action::BuildSettlement(color, *node_id))
                .collect()
        } else {
            panic!("Not implemented");
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
            .map(|edge_id| Action::BuildRoad(color, *edge_id))
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
                .map(|edge_id| Action::BuildRoad(color, *edge_id))
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
        if has_cities_available {
            return vec![];
        }

        self.get_settlements(color)
            .iter()
            .map(|building| match building {
                Building::Settlement(color, node_id) => Action::BuildCity(*color, *node_id),
                _ => panic!("Invalid building type"),
            })
            .collect()
    }

    pub fn play_turn_possibilities(&self, color: u8) -> Vec<Action> {
        if self.is_road_building() {
            return self.road_possibilities(color, true);
        } else if !self.current_player_rolled() {
            let mut actions = vec![Action::Roll(color)];
            if self.can_play_dev(DevCard::Knight as u8) {
                actions.push(Action::PlayKnight(color));
            }
            return actions;
        }

        let mut actions = vec![Action::EndTurn(color)];
        actions.extend(self.settlement_possibilities(color, false));
        actions.extend(self.road_possibilities(color, false));
        actions.extend(self.city_possibilities(color));

        if self.can_play_dev(DevCard::Knight as u8) {
            actions.push(Action::PlayKnight(color));
        }
        if self.can_play_dev(DevCard::YearOfPlenty as u8) {
            // TODO:
            // actions.push(Action::PlayYearOfPlenty(color));
        }
        if self.can_play_dev(DevCard::Monopoly as u8) {
            // TOOD:
            // actions.push(Action::PlayMonopoly(color));
        }
        if self.can_play_dev(DevCard::RoadBuilding as u8) {
            // TODO: What if user has no roads left? or is completely blocked?
            actions.push(Action::PlayRoadBuilding(color));
        }

        // TODO: Maritime trade possibilities

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_generation() {
        let state = State::new_base();
        let actions = state.generate_playable_actions();
        assert_eq!(actions.len(), 54);
        assert!(matches!(actions[0], Action::BuildSettlement(_, _)));
    }

    #[test]
    fn test_settlement_possibilities() {
        let state = State::new_base();

        let initial_build_phase_actions = state.settlement_possibilities(0, true);
        assert_eq!(initial_build_phase_actions.len(), 54);

        // TODO: Enable when implemented
        // let actions = state.settlement_possibilities(0, false);
        // assert_eq!(actions.len(), 3);
    }
}
