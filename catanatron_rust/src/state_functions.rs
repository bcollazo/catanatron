use crate::{enums::Action, state::State};

// ===== Mutable functions =====
pub fn apply_action(state: &mut State, action: Action) {
    println!("Applying action {:?} {:?}", state, action);
}
