use rand::prelude::*;

use crate::enums::Action;
use crate::state::State;
use super::Player;

pub struct RandomPlayer {}

impl Player for RandomPlayer {
    fn decide(&self, _state: &State, playable_actions: &[Action]) -> Action {
        let mut rng = rand::thread_rng();
        playable_actions
            .choose(&mut rng)
            .expect("There should always be at least one playable action")
            .clone()
    }
} 