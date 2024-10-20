use std::vec;

use crate::enums::ActionPrompt;
use crate::{
    actions::Action,
    enums::GameConfiguration,
    state_functions::{get_action_prompt, get_current_color},
    state_vector::StateVector,
};

pub fn generate_playable_actions(config: &GameConfiguration, state: &StateVector) -> Vec<Action> {
    println!("Generating playable actions");
    let current_color = get_current_color(config, state);
    let action_prompt = get_action_prompt(config, state);
    match action_prompt {
        ActionPrompt::BuildInitialSettlement => {
            settlement_possibilities(config, state, current_color, true)
        }
        ActionPrompt::BuildInitialRoad => todo!(),
        ActionPrompt::PlayTurn => todo!(),
        ActionPrompt::Discard => todo!(),
        ActionPrompt::MoveRobber => todo!(),
        // TODO:
        ActionPrompt::DecideTrade => todo!(),
        ActionPrompt::DecideAcceptees => todo!(),
    }
}

pub fn settlement_possibilities(
    config: &GameConfiguration,
    state: &StateVector,
    color: u8,
    is_initial_build_phase: bool,
) -> Vec<Action> {
    println!(
        "Generating settlement possibilities {:?} {:?} {:?} {:?}",
        config, state, color, is_initial_build_phase
    );
    // TODO: Actually read board and get buildable node ids to build actions with it
    vec![0, 1, 2]
}
