use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::PyResult;
use std::sync::Arc;
use std::fmt;
use rand::Rng;

use crate::players::Player as RustPlayer;
use crate::players::RandomPlayer as RustRandomPlayer;
use crate::state::State;
use crate::enums::Action as RustAction;
use super::action::{Action, rust_to_py_action};

/// Python-friendly Player class that can be subclassed in Python
#[pyclass(subclass)]
pub struct Player {
    /// The player's color (0 = RED, 1 = BLUE, 2 = ORANGE, 3 = WHITE)
    #[pyo3(get)]
    pub color: u8,
    
    /// Name of the player (optional)
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl Player {
    #[new]
    fn new(color: u8, name: Option<String>) -> Self {
        Self { 
            color, 
            name: name.unwrap_or_else(|| format!("Player {}", color)),
        }
    }
    
    /// Decide which action to take from the list of available actions
    ///
    /// This method should be overridden by Python subclasses.
    /// 
    /// Args:
    ///     state: The current game state
    ///     playable_actions: List of possible actions
    ///
    /// Returns:
    ///     The chosen action
    fn decide(&self, _py: Python, _state: PyObject, _playable_actions: Vec<Py<Action>>) -> PyResult<Py<Action>> {
        // In the base class, we just raise NotImplementedError to indicate
        // this should be implemented by subclasses
        Err(PyNotImplementedError::new_err(
            "Player.decide() must be implemented by a subclass"
        ))
    }
    
    fn __repr__(&self) -> String {
        format!("Player(color={}, name='{}')", self.color, self.name)
    }
    
    fn __str__(&self) -> String {
        if self.name.contains(&self.color.to_string()) {
            self.name.clone()
        } else {
            format!("{} ({})", self.name, self.color)
        }
    }
}

/// Python-friendly RandomPlayer implementation that uses the Rust backend
#[pyclass]
pub struct RandomPlayer {
    /// The player's color (0 = RED, 1 = BLUE, 2 = ORANGE, 3 = WHITE)
    #[pyo3(get)]
    pub color: u8,
    
    /// Name of the player
    #[pyo3(get, set)]
    pub name: String,
    
    /// Internal Rust player instance
    rust_player: RustRandomPlayer,
}

#[pymethods]
impl RandomPlayer {
    #[new]
    fn new(color: u8, name: Option<String>) -> Self {
        Self {
            color,
            name: name.unwrap_or_else(|| format!("RustRandom {}", color)),
            rust_player: RustRandomPlayer {},
        }
    }
    
    /// Decide which action to take using the Rust implementation
    fn decide(&self, py: Python, state: PyObject, playable_actions: Vec<Py<Action>>) -> PyResult<Py<Action>> {
        // Log the decision process to help with debugging
        println!("RustRandomPlayer.decide called with {} actions", playable_actions.len());
        
        if playable_actions.is_empty() {
            return Err(PyNotImplementedError::new_err(
                "No playable actions provided to RustRandomPlayer"
            ));
        }
        
        // Use Rust's random number generator to select an action
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..playable_actions.len());
        
        println!("RustRandomPlayer selected action index: {}", idx);
        
        // Return the selected action
        Ok(playable_actions[idx].clone())
    }
    
    fn __repr__(&self) -> String {
        format!("RustRandomPlayer(color={}, name='{}')", self.color, self.name)
    }
    
    fn __str__(&self) -> String {
        format!("RustRandom({})", self.name)
    }
}

/// Adapter that allows Python players to be used from Rust
pub struct PythonPlayerAdapter {
    py_player: PyObject,
    color: u8,
}

impl PythonPlayerAdapter {
    pub fn new(py_player: PyObject, color: u8) -> Self {
        Self { py_player, color }
    }
    
    /// Create a simple state dictionary for Python
    fn create_state_dict(&self, py: Python<'_>, state: &State) -> PyObject {
        let dict = PyDict::new(py);
        
        // Add basic game state information
        dict.set_item("current_color", state.get_current_color()).unwrap_or_default();
        dict.set_item("action_prompt", format!("{:?}", state.get_action_prompt())).unwrap_or_default();
        dict.set_item("is_initial_build_phase", state.is_initial_build_phase()).unwrap_or_default();
        dict.set_item("current_tick_seat", state.get_current_tick_seat()).unwrap_or_default();
        
        // Add player resource information for the current player
        let hand = state.get_player_hand(self.color);
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
        let dev_cards = state.get_player_devhand(self.color);
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
        dict.set_item("victory_points", state.get_actual_victory_points(self.color)).unwrap_or_default();
        
        // Add seating order
        if let Some(seating) = state.get_seating_order().get(0..state.get_seating_order().len()) {
            dict.set_item("seating_order", seating.to_vec()).unwrap_or_default();
        }
        
        dict.to_object(py)
    }
}

impl RustPlayer for PythonPlayerAdapter {
    fn decide(&self, state: &State, playable_actions: &[RustAction]) -> RustAction {
        Python::with_gil(|py| {
            // Convert state to Python
            let py_state = self.create_state_dict(py, state);
            
            // Convert actions to Python
            let py_actions = PyList::new(
                py,
                playable_actions.iter().map(|action| {
                    rust_to_py_action(py, action).unwrap_or_else(|_| {
                        // Fallback in case of conversion error
                        Py::new(py, Action {
                            action_type: "Unknown".to_string(),
                            params: PyDict::new(py).into(),
                            rust_action: *action,
                        }).unwrap()
                    })
                })
            );
            
            // Call Python player's decide method
            match self.py_player.call_method1(py, "decide", (py_state, py_actions)) {
                Ok(result) => {
                    // Try to extract the Action object
                    if let Ok(py_action) = result.extract::<&PyCell<Action>>(py) {
                        return py_action.borrow().rust_action;
                    }
                    
                    // Try to extract as index
                    if let Ok(idx) = result.extract::<usize>(py) {
                        if idx < playable_actions.len() {
                            return playable_actions[idx];
                        }
                    }
                    
                    // Try to extract as string and match action
                    if let Ok(action_str) = result.extract::<String>(py) {
                        for action in playable_actions {
                            if format!("{:?}", action) == action_str {
                                return *action;
                            }
                        }
                    }
                    
                    // Default to first action as fallback
                    playable_actions[0]
                },
                Err(e) => {
                    // Log the error and default to first action
                    eprintln!("Error calling Python player: {:?}", e);
                    playable_actions[0]
                }
            }
        })
    }
}

impl fmt::Debug for PythonPlayerAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PythonPlayerAdapter(color={})", self.color)
    }
}

/// Create a PythonPlayerAdapter from a PyObject
pub fn create_player_adapter(player: PyObject) -> PyResult<Arc<dyn RustPlayer>> {
    Python::with_gil(|py| {
        // Try to get the player's color - first look for _rust_color attribute
        let color = if let Ok(rust_color) = player.getattr(py, "_rust_color") {
            match rust_color.extract::<u8>(py) {
                Ok(color_val) => {
                    println!("Found _rust_color attribute: {}", color_val);
                    color_val
                },
                Err(e) => {
                    println!("Error extracting _rust_color: {:?}", e);
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("_rust_color attribute must be a u8 integer (0-255): {:?}", e)
                    ));
                }
            }
        } else if let Ok(py_player) = player.extract::<&PyCell<Player>>(py) {
            println!("Extracted player as PyCell<Player>");
            py_player.borrow().color
        } else if let Ok(color_attr) = player.getattr(py, "color") {
            println!("Found color attribute, attempting to extract");
            
            // Try to extract as u8 first
            if let Ok(color_val) = color_attr.extract::<u8>(py) {
                println!("  - Extracted as u8: {}", color_val);
                color_val
            } else {
                // If that fails, try to see if it's an enum with a 'name' attribute
                if let Ok(name_attr) = color_attr.getattr(py, "name") {
                    if let Ok(name_str) = name_attr.extract::<String>(py) {
                        println!("  - Found color.name: {}", name_str);
                        // Map the color name to a u8
                        match name_str.as_str() {
                            "RED" => 0,
                            "BLUE" => 1,
                            "ORANGE" => 2,
                            "WHITE" => 3,
                            _ => {
                                println!("  - Unknown color name: {}", name_str);
                                return Err(pyo3::exceptions::PyValueError::new_err(
                                    format!("Unknown color name: {}", name_str)
                                ));
                            }
                        }
                    } else {
                        println!("  - color.name is not a string");
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "color.name is not a string"
                        ));
                    }
                } else {
                    println!("  - color attribute is not a u8 or an enum with name");
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "color attribute must be a u8 integer or an enum with a 'name' attribute"
                    ));
                }
            }
        } else {
            println!("Player object has no color or _rust_color attribute");
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Player object must have a 'color' attribute or '_rust_color' attribute"
            ));
        };
        
        println!("Creating PythonPlayerAdapter with color={}", color);
        Ok(Arc::new(PythonPlayerAdapter::new(player, color)) as Arc<dyn RustPlayer>)
    })
} 