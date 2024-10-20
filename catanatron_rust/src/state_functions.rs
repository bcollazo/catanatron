use crate::enums::GameConfiguration;
use crate::state::{actual_victory_points_index, seating_order_slice, State, CURRENT_PLAYER_INDEX};

pub fn get_current_color(config: &GameConfiguration, state: &State) -> u8 {
    let current_player_index = state[CURRENT_PLAYER_INDEX];
    let color_seating_order = &state[seating_order_slice(config.num_players as usize)];
    color_seating_order[current_player_index as usize]
}

pub fn winner(config: &GameConfiguration, state: &State) -> Option<u8> {
    let current_color = get_current_color(config, state);

    let actual_victory_points =
        state[actual_victory_points_index(config.num_players, current_color)];
    if actual_victory_points >= config.vps_to_win {
        return Some(current_color);
    }
    None
}
