use rand::Rng;

use crate::{enums::Action, state::State};

pub trait Player {
    fn decide(&self, state: &State, playable_actions: &[Action]) -> Action;
}

pub struct RandomPlayer {}

impl Player for RandomPlayer {
    fn decide(&self, _state: &State, playable_actions: &[Action]) -> Action {
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..playable_actions.len());
        playable_actions[index]
    }
}
