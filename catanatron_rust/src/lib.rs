use log::info;
use pyo3::prelude::*;
use std::sync::Once;

pub mod deck_slices;
pub mod decks;
pub mod enums;
pub mod game;
pub mod global_state;
pub mod map_instance;
pub mod map_template;
mod ordered_hashmap;
pub mod players;
pub mod state;
pub mod state_vector;
pub mod python;

// Ensure logger is initialized only once for Python module
static PYTHON_LOGGER_INIT: Once = Once::new();

#[pymodule]
fn catanatron_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    PYTHON_LOGGER_INIT.call_once(|| {
        env_logger::init();
        info!("Initialized catanatron_rust logging");
    });
    
    // Export the Game class at top level
    m.add_class::<game::Game>()?;
    
    // Also export the Python-friendly Game class at the top level for easier access
    m.add_class::<python::Game>()?;
    
    // Create a 'python' submodule
    let python_module = PyModule::new(_py, "python")?;
    
    // Add Python-friendly classes to the python submodule
    python_module.add_class::<python::Action>()?;
    python_module.add_class::<python::Player>()?;
    python_module.add_class::<python::RandomPlayer>()?;
    python_module.add_class::<python::Game>()?;
    
    // Add utility functions to the python submodule
    python_module.add_function(wrap_pyfunction!(python::game::create_game, _py)?)?;
    
    // Add the python submodule to the main module
    m.add_submodule(python_module)?;
    
    Ok(())
}
