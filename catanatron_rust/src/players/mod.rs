mod random_player;

use crate::{enums::Action, state::State};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use log::{debug, warn};

pub use random_player::RandomPlayer;

pub trait Player {
    fn decide(&self, state: &State, playable_actions: &[Action]) -> Action;
}

pub struct PythonPlayerWrapper {
    player_obj: PyObject,
}

impl PythonPlayerWrapper {
    pub fn new(player_obj: PyObject) -> Self {
        Self { player_obj }
    }
    
    // Create a simplified state representation for Python
    fn create_state_dict(&self, py: Python<'_>, state: &State) -> PyObject {
        let dict = PyDict::new(py);
        
        // Add basic game state information
        dict.set_item("current_color", state.get_current_color()).unwrap_or_default();
        dict.set_item("action_prompt", format!("{:?}", state.get_action_prompt())).unwrap_or_default();
        dict.set_item("is_initial_build_phase", state.is_initial_build_phase()).unwrap_or_default();
        dict.set_item("current_tick_seat", state.get_current_tick_seat()).unwrap_or_default();
        
        // Add player resource information for the current player
        let current_color = state.get_current_color();
        let hand = state.get_player_hand(current_color);
        if hand.len() >= 5 {
            let resources = PyDict::new(py);
            resources.set_item("wood", hand[0]).unwrap_or_default();
            resources.set_item("brick", hand[1]).unwrap_or_default();
            resources.set_item("sheep", hand[2]).unwrap_or_default();
            resources.set_item("wheat", hand[3]).unwrap_or_default();
            resources.set_item("ore", hand[4]).unwrap_or_default();
            dict.set_item("resources", resources).unwrap_or_default();
        }
        
        // Add player development cards
        let dev_cards = state.get_player_devhand(current_color);
        if dev_cards.len() >= 5 {
            let cards = PyDict::new(py);
            cards.set_item("knight", dev_cards[0]).unwrap_or_default();
            cards.set_item("year_of_plenty", dev_cards[1]).unwrap_or_default();
            cards.set_item("monopoly", dev_cards[2]).unwrap_or_default();
            cards.set_item("road_building", dev_cards[3]).unwrap_or_default();
            cards.set_item("victory_point", dev_cards[4]).unwrap_or_default();
            dict.set_item("dev_cards", cards).unwrap_or_default();
        }
        
        // Add victory points information
        dict.set_item("victory_points", state.get_actual_victory_points(current_color)).unwrap_or_default();
        
        // Add seating order
        if let Some(seating) = state.get_seating_order().get(0..state.get_seating_order().len()) {
            dict.set_item("seating_order", seating.to_vec()).unwrap_or_default();
        }
        
        dict.to_object(py)
    }
}

impl Player for PythonPlayerWrapper {
    fn decide(&self, state: &State, playable_actions: &[Action]) -> Action {
        Python::with_gil(|py| {
            // Convert actions to Python list of Action objects
            let py_actions = PyList::empty(py);
            for action in playable_actions.iter() {
                let py_action = Python::with_gil(|py| {
                    // Convert Rust Action to Python Action using string representation
                    let action_str = format!("{:?}", action);
                    let params = PyDict::new(py);
                    params.set_item("action", action_str).unwrap_or_default();
                    params.to_object(py)
                });
                py_actions.append(py_action).unwrap_or_default();
            }
            
            // Create a simplified state representation for Python
            let py_state = self.create_state_dict(py, state);
            
            debug!("Calling Python player's decide method with {} actions", playable_actions.len());
            
            // First try the new approach with state parameter
            let result = match self.player_obj.call_method1(py, "decide", (py_state, py_actions)) {
                Ok(result) => {
                    debug!("Successfully called Python player with state information");
                    result
                },
                Err(e) => {
                    // Fall back to the original approach with just actions
                    debug!("Error calling Python player with state, falling back: {:?}", e);
                    match self.player_obj.call_method1(py, "decide", (py_actions,)) {
                        Ok(result) => {
                            debug!("Successfully called Python player with just actions");
                            result
                        },
                        Err(e) => {
                            warn!("Error calling Python player's decide method: {:?}", e);
                            // Default to the first action as a fallback
                            return playable_actions[0];
                        }
                    }
                }
            };
            
            // Convert the Python result back to a Rust Action
            match result.extract::<usize>(py) {
                Ok(idx) => {
                    if idx < playable_actions.len() {
                        playable_actions[idx]
                    } else {
                        warn!("Python player returned invalid action index: {}", idx);
                        playable_actions[0] // Default to first action
                    }
                },
                Err(_) => {
                    // Try to extract as string (format of Action::Variant {...})
                    match result.extract::<String>(py) {
                        Ok(action_str) => {
                            // Find matching action based on string representation
                            for action in playable_actions.iter() {
                                if format!("{:?}", action) == action_str {
                                    return *action;
                                }
                            }
                            // Default to first action if no match found
                            playable_actions[0]
                        },
                        Err(_) => {
                            warn!("Could not convert Python result to action");
                            playable_actions[0]
                        }
                    }
                }
            }
        })
    }
}
