use std::vec;

use crate::actions::Action;
use crate::enums::ActionPrompt;
use crate::state::State;

pub fn generate_playable_actions(state: &State) -> Vec<Action> {
    println!("Generating playable actions");
    let current_color = state.get_current_color();
    let action_prompt = state.get_action_prompt();
    match action_prompt {
        ActionPrompt::BuildInitialSettlement => {
            settlement_possibilities(state, current_color, true)
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
    state: &State,
    color: u8,
    is_initial_build_phase: bool,
) -> Vec<Action> {
    println!(
        "Generating settlement possibilities {:?} {:?} {:?}",
        state, color, is_initial_build_phase
    );
    // TODO: Actually read board and get buildable node ids to build actions with it
    vec![0, 1, 2]
}
