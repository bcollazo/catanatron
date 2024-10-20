use std::collections::HashMap;

use crate::enums::GameConfiguration;
use crate::move_generation::generate_playable_actions;
use crate::player::Player;
use crate::state::{initialize_state, State};
use crate::state_functions::{apply_action, get_current_color, winner};

pub fn play_game(config: GameConfiguration, players: HashMap<u8, Box<dyn Player>>) -> Option<u8> {
    println!("Playing game with configuration: {:?}", config);
    let mut state = initialize_state();
    let mut num_turns = 0;
    while winner(&config, &state).is_none() && num_turns < config.max_turns {
        play_tick(&config, &players, &mut state);
        num_turns += 1;
    }
    winner(&config, &state)
}

fn play_tick(
    config: &GameConfiguration,
    players: &HashMap<u8, Box<dyn Player>>,
    state: &mut State,
) {
    println!("Playing config {:?}", config);
    println!("Playing turn {:?}", state);
    let current_color = get_current_color(config, state);
    let current_player = players.get(&current_color).unwrap();

    let playable_actions = generate_playable_actions(config, state);
    let action = current_player.decide(state, &playable_actions);
    println!(
        "Player {:?} decided to play action {:?}",
        current_color, action
    );

    apply_action(config, state, action);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{enums::MapType, player::RandomPlayer};

    #[test]
    fn test_game_creation() {
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

        let result = play_game(config, players);
        assert_eq!(result, None);
    }
}
