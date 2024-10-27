use std::collections::HashMap;
use std::rc::Rc;

use crate::enums::GameConfiguration;
use crate::global_state::GlobalState;
use crate::map_instance::MapInstance;
use crate::player::Player;
use crate::state::State;

pub fn play_game(
    global_state: GlobalState,
    config: GameConfiguration,
    players: HashMap<u8, Box<dyn Player>>,
) -> Option<u8> {
    let map_instance = MapInstance::new(
        &global_state.base_map_template,
        &global_state.dice_probas,
        0,
    );
    let rc_config = Rc::new(config);
    println!("Playing game with configuration: {:?}", rc_config);
    let mut state = State::new(rc_config.clone(), Rc::new(map_instance));
    println!("Seat order: {:?}", state.get_seating_order());
    let mut num_ticks = 0;
    while state.winner().is_none() && num_ticks < rc_config.max_ticks {
        println!("Tick {:?} =====", num_ticks);
        play_tick(&players, &mut state);
        num_ticks += 1;
    }
    state.winner()
}

fn play_tick(players: &HashMap<u8, Box<dyn Player>>, state: &mut State) {
    let current_color = state.get_current_color();
    let current_player = players.get(&current_color).unwrap();

    let playable_actions = state.generate_playable_actions();
    println!(
        "Player {:?} has {:?} playable actions",
        current_color, playable_actions
    );
    let action = current_player.decide(state, &playable_actions);
    println!(
        "Player {:?} decided to play action {:?}",
        current_color, action
    );

    state.apply_action(action);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        enums::{Action, ActionPrompt, MapType},
        player::RandomPlayer,
    };

    fn setup_game() -> (GlobalState, GameConfiguration, HashMap<u8, Box<dyn Player>>) {
        let global_state = GlobalState::new();
        let config = GameConfiguration {
            dicard_limit: 7,
            vps_to_win: 10,
            map_type: MapType::Base,
            num_players: 4,
            max_ticks: 8, // TODO: Change!
        };
        let mut players: HashMap<u8, Box<dyn Player>> = HashMap::new();
        players.insert(0, Box::new(RandomPlayer {}));
        players.insert(1, Box::new(RandomPlayer {}));
        players.insert(2, Box::new(RandomPlayer {}));
        players.insert(3, Box::new(RandomPlayer {}));
        (global_state, config, players)
    }

    #[test]
    fn test_game_creation() {
        let (global_state, config, players) = setup_game();

        let result = play_game(global_state, config, players);
        assert_eq!(result, None);
    }

    #[test]
    fn test_initial_build_phase() {
        let (global_state, config, players) = setup_game();
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );
        let rc_config = Rc::new(config);
        let mut state = State::new(rc_config.clone(), Rc::new(map_instance));

        let seating_order = state.get_seating_order();
        let first_player = seating_order[0];
        let second_player = seating_order[1];
        let third_player = seating_order[2];
        let fourth_player = seating_order[3];

        // first player settlement
        let playable_actions = state.generate_playable_actions();
        assert_eq!(playable_actions.len(), 54);
        assert_all_build_settlements(playable_actions, first_player);
        play_tick(&players, &mut state);

        // first player road
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 2);
        assert_all_build_roads(playable_actions, first_player);
        assert!(state.is_initial_build_phase());
        play_tick(&players, &mut state);

        // second player settlement: assert at 50-51 actions and all are build settlement
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 50 && playable_actions.len() <= 51);
        assert_all_build_settlements(playable_actions, second_player);
        play_tick(&players, &mut state);

        // second player road: assert at least 2 actions and all are build road
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 2);
        assert_all_build_roads(playable_actions, second_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // third player settlement

        // third player road
        let playable_actions = state.generate_playable_actions();
        assert_all_build_roads(playable_actions, third_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // fourth player settlement
        play_tick(&players, &mut state); // fourth player road

        // fourth player settlement 2
        assert!(state.is_initial_build_phase());
        let playable_actions = state.generate_playable_actions();
        assert_all_build_settlements(playable_actions, fourth_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // fourth player road
        play_tick(&players, &mut state); // third player settlement 2
        play_tick(&players, &mut state); // third player road
        play_tick(&players, &mut state);

        // second player road 2
        let playable_actions = state.generate_playable_actions();
        assert_all_build_roads(playable_actions, second_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // first player settlement 2
        play_tick(&players, &mut state); // first player road

        // Assert that the initial build phase is over and its the first player's turn
        assert!(!state.is_initial_build_phase());
        assert_eq!(state.get_current_color(), first_player);
        assert!(matches!(state.get_action_prompt(), ActionPrompt::PlayTurn));

        // TODO: Assert players have money of their second house
    }

    fn assert_all_build_settlements(playable_actions: Vec<Action>, player: u8) {
        assert!(
            playable_actions.iter().all(|e| {
                if let Action::BuildSettlement(p, _) = e {
                    *p == player
                } else {
                    false
                }
            }),
            "Expected all actions to be BuildSettlement for player {:?}",
            player
        );
    }

    fn assert_all_build_roads(playable_actions: Vec<Action>, player: u8) {
        assert!(
            playable_actions.iter().all(|e| {
                if let Action::BuildRoad(p, _) = e {
                    *p == player
                } else {
                    false
                }
            }),
            "Expected all actions to be BuildRoad for player {:?}",
            player
        );
    }
}
