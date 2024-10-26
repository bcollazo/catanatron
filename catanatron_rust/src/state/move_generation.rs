use crate::deck_slices::{freqdeck_contains, ROAD_COST};
use crate::enums::{Action, ActionPrompt};
use crate::state::State;

const TOTAL_ROADS_PER_PLAYER: u8 = 15;

impl State {
    pub fn generate_playable_actions(&self) -> Vec<Action> {
        println!("Generating playable actions");
        let current_color = self.get_current_color();
        let action_prompt = self.get_action_prompt();
        match action_prompt {
            ActionPrompt::BuildInitialSettlement => {
                self.settlement_possibilities(current_color, true)
            }
            ActionPrompt::BuildInitialRoad => self.road_possibilities(current_color, true),
            ActionPrompt::PlayTurn => todo!(),
            ActionPrompt::Discard => todo!(),
            ActionPrompt::MoveRobber => todo!(),
            // TODO:
            ActionPrompt::DecideTrade => todo!(),
            ActionPrompt::DecideAcceptees => todo!(),
        }
    }

    pub fn settlement_possibilities(&self, color: u8, is_initial_build_phase: bool) -> Vec<Action> {
        println!(
            "Generating settlement possibilities {:?} {:?}",
            color, is_initial_build_phase
        );
        if is_initial_build_phase {
            self.board_buildable_ids
                .iter()
                .map(|node_id| Action::BuildSettlement(color, *node_id))
                .collect()
        } else {
            panic!("Not implemented");
        }
    }

    pub fn road_possibilities(&self, color: u8, is_free: bool) -> Vec<Action> {
        println!("Generating road possibilities {:?} {:?}", color, is_free);
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
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::enums::{GameConfiguration, MapType};
    use crate::global_state::GlobalState;
    use crate::map_instance::MapInstance;

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
    fn test_move_generation() {
        let state = setup_state();
        let actions = state.generate_playable_actions();
        assert_eq!(actions.len(), 54);
        assert!(matches!(actions[0], Action::BuildSettlement(_, _)));
    }

    #[test]
    fn test_settlement_possibilities() {
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
        let state = State::new(Rc::new(config), Rc::new(map_instance));

        let initial_build_phase_actions = state.settlement_possibilities(0, true);
        assert_eq!(initial_build_phase_actions.len(), 54);

        // TODO: Enable when implemented
        // let actions = state.settlement_possibilities(0, false);
        // assert_eq!(actions.len(), 3);
    }
}
