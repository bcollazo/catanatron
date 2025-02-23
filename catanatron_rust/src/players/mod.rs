mod random_player;

use crate::enums::Action;
use crate::state::State;

pub trait Player {
    fn decide(&self, state: &State, playable_actions: &[Action]) -> Action;
}

pub use random_player::RandomPlayer; 