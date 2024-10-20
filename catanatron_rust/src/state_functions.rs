use crate::enums::{ActionPrompt, GameConfiguration};
use crate::state_vector::{
    actual_victory_points_index, seating_order_slice, StateVector, CURRENT_TICK_SEAT_INDEX,
    IS_DISCARDING_INDEX, IS_INITIAL_BUILD_PHASE_INDEX, IS_MOVING_ROBBER_INDEX,
};

// ===== Read-only functions =====
pub fn is_initial_build_phase(state: &StateVector) -> bool {
    state[IS_INITIAL_BUILD_PHASE_INDEX] == 1
}

pub fn is_moving_robber(state: &StateVector) -> bool {
    state[IS_MOVING_ROBBER_INDEX] == 1
}

pub fn is_discarding(state: &StateVector) -> bool {
    state[IS_DISCARDING_INDEX] == 1
}

pub fn get_current_color(config: &GameConfiguration, state: &StateVector) -> u8 {
    let seating_order = get_seating_order(config, state);
    let current_tick_seat = state[CURRENT_TICK_SEAT_INDEX];
    seating_order[current_tick_seat as usize]
}

/// Returns a slice of Colors in the order of seating
/// e.g. [2, 1, 0, 3] if Orange goes first, then Blue, then Red, and then White
fn get_seating_order<'a>(config: &GameConfiguration, state: &'a [u8]) -> &'a [u8] {
    &state[seating_order_slice(config.num_players as usize)]
}

pub fn get_action_prompt(config: &GameConfiguration, state: &StateVector) -> ActionPrompt {
    if is_initial_build_phase(state) {
        let num_things_built = 0;
        if num_things_built == 2 * config.num_players {
            return ActionPrompt::PlayTurn;
        } else if num_things_built % 2 == 0 {
            return ActionPrompt::BuildInitialSettlement;
        } else {
            return ActionPrompt::BuildInitialRoad;
        }
    } else if is_moving_robber(state) {
        return ActionPrompt::MoveRobber;
    } else if is_discarding(state) {
        return ActionPrompt::Discard;
    } // TODO: Implement Trading Prompts (DecideTrade, DecideAcceptees)
    ActionPrompt::PlayTurn
}

pub fn winner(config: &GameConfiguration, state: &StateVector) -> Option<u8> {
    let current_color = get_current_color(config, state);

    let actual_victory_points =
        state[actual_victory_points_index(config.num_players, current_color)];
    if actual_victory_points >= config.vps_to_win {
        return Some(current_color);
    }
    None
}

// ===== Mutable functions =====
pub fn apply_action(config: &GameConfiguration, state: &mut StateVector, action: u64) {
    println!("Applying action {:?} {:?} {:?}", config, state, action);
}
