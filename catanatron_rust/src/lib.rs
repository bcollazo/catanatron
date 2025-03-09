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
    
    // Export the Game class
    m.add_class::<game::Game>()?;
    
    // Export Python-friendly classes
    m.add_class::<python::Action>()?;
    m.add_class::<python::Player>()?;
    m.add_class::<python::RandomPlayer>()?;
    m.add_class::<python::Game>()?;
    
    // Export utility functions
    m.add_function(wrap_pyfunction!(python::game::create_game, m)?)?;
    
    // Export enums for use in Python
    // In a future version, we'll want to export more types to match the Python API
    
    Ok(())
}
