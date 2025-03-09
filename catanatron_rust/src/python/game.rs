use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

use crate::global_state::GlobalState;
use crate::map_instance::MapInstance;
use crate::state::State;
use crate::players::Player as RustPlayer;
use crate::enums::{GameConfiguration, MapType};

use super::state::rust_state_to_py_dict;
use super::player::create_player_adapter;
use super::action::{Action, rust_to_py_action};

/// Python-friendly Game class
#[pyclass(unsendable)]
pub struct Game {
    /// Game configuration
    config: GameConfiguration,
    
    /// Player objects (keeps Python references alive)
    players: Vec<PyObject>,
    
    /// Random seed (if specified)
    seed: Option<i64>,
    
    /// Internal state of the game
    state: Option<State>,
    
    /// Map instance
    map_instance: Option<Arc<MapInstance>>,
    
    /// Player adapters
    rust_players: HashMap<u8, Arc<dyn RustPlayer>>,
    
    /// Global state (singleton)
    global_state: GlobalState,
    
    /// Winner color (if any)
    winner: Option<u8>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (players, seed=None, discard_limit=7, vps_to_win=10, map_type="BASE"))]
    fn new(
        py: Python,
        players: &PyList,
        seed: Option<i64>,
        discard_limit: u8,
        vps_to_win: u8,
        map_type: &str,
    ) -> PyResult<Self> {
        // Validate player count
        let num_players = players.len();
        if !(2..=4).contains(&num_players) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Game requires 2-4 players, got {}", num_players)
            ));
        }
        
        // Parse map type
        let map_type = match map_type.to_uppercase().as_str() {
            "BASE" => MapType::Base,
            "MINI" => MapType::Mini,
            "TOURNAMENT" => MapType::Tournament,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown map type: {}", map_type)
                ));
            }
        };
        
        // Create configuration
        let config = GameConfiguration {
            discard_limit,
            vps_to_win,
            map_type,
            num_players: num_players as u8,
            max_ticks: 10000, // Reasonable default
        };
        
        // Store player references
        let player_refs: Vec<PyObject> = players.iter().map(|p| p.into()).collect();
        
        // Create player adapters
        let mut rust_players = HashMap::new();
        for (i, player) in player_refs.iter().enumerate() {
            let player_adapter = create_player_adapter(player.clone_ref(py))?;
            rust_players.insert(i as u8, player_adapter);
        }
        
        // Create global state
        let global_state = GlobalState::new();
        
        // Create the Game instance without initializing
        Ok(Self {
            config,
            players: player_refs,
            seed,
            state: None,
            map_instance: None,
            rust_players,
            global_state,
            winner: None,
        })
    }
    
    /// Play the game until completion
    ///
    /// Args:
    ///     accumulators: Optional list of accumulator objects to track game state
    ///
    /// Returns:
    ///     The color of the winning player, or None if no winner
    fn play(&mut self, py: Python, accumulators: Option<Vec<PyObject>>) -> PyResult<Option<u8>> {
        // Initialize the game if not already initialized
        if self.state.is_none() {
            self.initialize(py)?;
        }
        
        let state = self.state.as_mut().unwrap();
        
        // Play until we have a winner
        while self.winner.is_none() {
            // Call accumulators before step
            if let Some(accs) = &accumulators {
                for acc in accs {
                    let state_dict = rust_state_to_py_dict(py, state)?;
                    let _ = acc.call_method1(py, "before", (state_dict,));
                }
            }
            
            // Generate playable actions
            let playable_actions = state.generate_playable_actions();
            
            // Get current color and decide on an action
            let current_color = state.get_current_color();
            let player = self.rust_players.get(&current_color).unwrap();
            let action = player.decide(state, &playable_actions);
            
            // Apply the action to the state
            state.apply_action(action);
            
            // Call accumulators after step
            if let Some(accs) = &accumulators {
                for acc in accs {
                    let state_dict = rust_state_to_py_dict(py, state)?;
                    let py_action = rust_to_py_action(py, &action)?;
                    let _ = acc.call_method1(py, "step", (state_dict, py_action));
                }
            }
            
            // Check for a winner
            self.winner = state.winner();
        }
        
        // Call accumulators after game end
        if let Some(accs) = &accumulators {
            for acc in accs {
                let state_dict = rust_state_to_py_dict(py, state)?;
                let _ = acc.call_method1(py, "after", (state_dict,));
            }
        }
        
        Ok(self.winner)
    }
    
    /// Play a single tick of the game
    ///
    /// Returns:
    ///     The action taken in this tick
    fn play_tick(&mut self, py: Python) -> PyResult<Py<Action>> {
        // Initialize the game if not already initialized
        if self.state.is_none() {
            self.initialize(py)?;
        }
        
        let state = self.state.as_mut().unwrap();
        
        // Generate playable actions
        let playable_actions = state.generate_playable_actions();
        
        // Get current color and decide on an action
        let current_color = state.get_current_color();
        let player = self.rust_players.get(&current_color).unwrap();
        let action = player.decide(state, &playable_actions);
        
        // Apply the action to the state
        state.apply_action(action);
        
        // Update winner status
        self.winner = state.winner();
        
        // Convert to Python action
        rust_to_py_action(py, &action)
    }
    
    /// Get the number of players in the game
    fn get_num_players(&self) -> usize {
        self.players.len()
    }
    
    /// Get the winner of the game
    fn get_winner(&self) -> Option<u8> {
        self.winner
    }
    
    /// Get a string representation of the game state
    fn get_state_repr(&self, py: Python) -> PyResult<PyObject> {
        if let Some(state) = &self.state {
            let dict = rust_state_to_py_dict(py, state)?;
            Ok(dict.into())
        } else {
            Ok(PyDict::new(py).into())
        }
    }
    
    /// Initialize the game
    fn initialize(&mut self, _py: Python) -> PyResult<()> {
        // Create MapInstance using global_state
        let map_instance = MapInstance::new(
            &self.global_state.base_map_template,
            &self.global_state.dice_probas,
            self.seed.map(|s| s as u64).unwrap_or(0),
        );
        
        // Create and initialize game state - now with Clone
        let map_instance_rc = std::rc::Rc::new(map_instance.clone());
        let state = State::new(
            std::rc::Rc::new(self.config.clone()),
            map_instance_rc,
        );
        
        self.state = Some(state);
        self.map_instance = Some(Arc::new(map_instance));
        
        Ok(())
    }

    /// Get the current player's color
    fn get_current_player(&self) -> PyResult<u8> {
        if let Some(state) = &self.state {
            Ok(state.get_current_color())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }

    /// Get the current action prompt
    fn get_action_prompt(&self) -> PyResult<String> {
        if let Some(state) = &self.state {
            Ok(format!("{:?}", state.get_action_prompt()))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }

    /// Get the current playable actions
    fn get_playable_actions(&self, py: Python) -> PyResult<Py<PyList>> {
        if let Some(state) = &self.state {
            let playable_actions = state.generate_playable_actions();
            let actions_list = PyList::empty(py);
            
            for action in playable_actions {
                let py_action = rust_to_py_action(py, &action)?;
                actions_list.append(py_action)?;
            }
            
            Ok(actions_list.into())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }

    /// Check if the game is in the initial build phase
    fn is_initial_build_phase(&self) -> PyResult<bool> {
        if let Some(state) = &self.state {
            Ok(state.is_initial_build_phase())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }

    /// Get player's victory points
    fn get_victory_points(&self, color: u8) -> PyResult<u8> {
        if let Some(state) = &self.state {
            Ok(state.get_actual_victory_points(color))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }

    /// Get player's resources
    fn get_resources(&self, py: Python, color: u8) -> PyResult<Py<PyDict>> {
        if let Some(state) = &self.state {
            let hand = state.get_player_hand(color);
            let resources = PyDict::new(py);
            
            if hand.len() >= 5 {
                resources.set_item("wood", hand[0])?;
                resources.set_item("brick", hand[1])?;
                resources.set_item("sheep", hand[2])?;
                resources.set_item("wheat", hand[3])?;
                resources.set_item("ore", hand[4])?;
            }
            
            Ok(resources.into())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }

    /// Apply a custom action to the game state
    fn apply_action(&mut self, py: Python, action: Py<Action>) -> PyResult<()> {
        if let Some(state) = &mut self.state {
            // Extract the action and convert to Rust action
            let action_obj = action.extract::<Action>(py)?;
            let rust_action = action_obj.rust_action;
            
            // Apply action
            state.apply_action(rust_action);
            
            // Update winner status
            self.winner = state.winner();
            
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Game not initialized"
            ))
        }
    }
}

/// Create a new Game instance with the specified configuration
#[pyfunction]
#[pyo3(signature = (players, seed=None, discard_limit=7, vps_to_win=10, map_type="BASE"))]
pub fn create_game(
    py: Python,
    players: &PyList,
    seed: Option<i64>,
    discard_limit: u8,
    vps_to_win: u8,
    map_type: &str,
) -> PyResult<Game> {
    Game::new(py, players, seed, discard_limit, vps_to_win, map_type)
} 