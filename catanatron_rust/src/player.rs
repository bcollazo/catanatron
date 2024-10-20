use rand::Rng;

use crate::actions::Action;
use crate::state::State;

pub trait Player {
    fn decide(&self, state: &State, playable_actions: &Vec<Action>) -> u64;
}

pub struct RandomPlayer {}

impl Player for RandomPlayer {
    fn decide(&self, _state: &State, playable_actions: &Vec<Action>) -> u64 {
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..playable_actions.len());
        playable_actions[index]
    }
}
