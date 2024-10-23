use crate::enums::{ActionPrompt, GameConfiguration};
use crate::state::State;

// ===== Mutable functions =====
pub fn apply_action(state: &mut State, action: u64) {
    println!("Applying action {:?} {:?}", state, action);
}
