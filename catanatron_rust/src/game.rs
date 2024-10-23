use std::collections::HashMap;
use std::rc::Rc;

use crate::enums::GameConfiguration;
use crate::global_state::GlobalState;
use crate::map_instance::MapInstance;
use crate::player::Player;
use crate::state::State;
use crate::state_functions::apply_action;

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
        play_tick(&players, &mut state);
        num_turns += 1;
    }
    state.winner()
}

fn play_tick(players: &HashMap<u8, Box<dyn Player>>, state: &mut State) {
    println!("Playing turn {:?}", state);
    let current_color = state.get_current_color();
    let current_player = players.get(&current_color).unwrap();

    let playable_actions = state.generate_playable_actions();
    let action = current_player.decide(state, &playable_actions);
    println!(
        "Player {:?} decided to play action {:?}",
        current_color, action
    );

    apply_action(state, action);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{enums::MapType, player::RandomPlayer};

    #[test]
    fn test_game_creation() {
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

        let result = play_game(global_state, config, players);
        assert_eq!(result, None);
    }
}
