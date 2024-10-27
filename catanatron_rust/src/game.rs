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
    let mut num_ticks = 0;
    while state.winner().is_none() && num_ticks < rc_config.max_ticks {
        println!("Playing turn {:?}", num_ticks);
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
        enums::{Action, MapType},
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

        let first_player = state.get_current_color();

        let playable_actions = state.generate_playable_actions();
        assert_eq!(playable_actions.len(), 54);
        assert!(playable_actions.iter().all(|e| {
            if let Action::BuildSettlement(player, _) = e {
                *player == first_player
            } else {
                false
            }
        }));

        play_tick(&players, &mut state);

        // assert at least 2 actions and all are build road
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 2);
        assert!(playable_actions
            .iter()
            .all(|e| matches!(e, Action::BuildRoad(_, _))));
        assert!(state.is_initial_build_phase());

        play_tick(&players, &mut state);

        // assert at 50 actions and all are build settlement
        let playable_actions = state.generate_playable_actions();
        assert_eq!(playable_actions.len(), 50);
        assert!(playable_actions
            .iter()
            .all(|e| matches!(e, Action::BuildSettlement(_, _))));
    }
}
