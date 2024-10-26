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
    let mut num_turns = 0;
    while state.winner().is_none() && num_turns < rc_config.max_turns {
        println!("Playing turn {:?}", num_turns);
        play_tick(&players, &mut state);
        num_turns += 1;
    }
    state.winner()
}

fn play_tick(players: &HashMap<u8, Box<dyn Player>>, state: &mut State) {
    let current_color = state.get_current_color();
    let current_player = players.get(&current_color).unwrap();

    let playable_actions = state.generate_playable_actions();
    println!(
        "Player {:?} has {:?} playable actions",
        current_color,
        playable_actions.len()
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
    use crate::{enums::MapType, player::RandomPlayer};

    fn setup_game() -> (GlobalState, GameConfiguration, HashMap<u8, Box<dyn Player>>) {
        let global_state = GlobalState::new();
        let config = GameConfiguration {
            dicard_limit: 7,
            vps_to_win: 10,
            map_type: MapType::Base,
            num_players: 4,
            max_turns: 100,
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

        assert_eq!(state.generate_playable_actions().len(), 54);
        play_tick(&players, &mut state);
        assert!(state.is_initial_build_phase());
        assert_eq!(state.generate_playable_actions().len(), 3);
    }
}
