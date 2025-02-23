use pyo3::prelude::*;

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

#[pymodule]
fn catanatron_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<game::Game>()?;
    Ok(())
}
