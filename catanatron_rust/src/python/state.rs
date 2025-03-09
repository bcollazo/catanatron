use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::state::State as RustState;

/// Convert a color (u8) to a string representation
fn color_to_string(color: u8) -> &'static str {
    match color {
        0 => "RED",
        1 => "BLUE",
        2 => "ORANGE",
        3 => "WHITE",
        _ => "UNKNOWN",
    }
}

/// Convert a Rust State object to a Python dictionary representation
pub fn rust_state_to_py_dict(py: Python, state: &RustState) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    
    // Add basic state information
    dict.set_item("current_player", color_to_string(state.get_current_color()))?;
    dict.set_item("current_player_color", state.get_current_color())?;
    dict.set_item("is_initial_build_phase", state.is_initial_build_phase())?;
    dict.set_item("current_tick_seat", state.get_current_tick_seat())?;
    dict.set_item("action_prompt", format!("{:?}", state.get_action_prompt()))?;
    dict.set_item("has_rolled", state.current_player_rolled())?;
    
    // Add bank information
    let bank_dict = PyDict::new(py);
    let bank_resources = state.get_bank_resources();
    if bank_resources.len() >= 5 {
        bank_dict.set_item("wood", bank_resources[0])?;
        bank_dict.set_item("brick", bank_resources[1])?;
        bank_dict.set_item("sheep", bank_resources[2])?;
        bank_dict.set_item("wheat", bank_resources[3])?;
        bank_dict.set_item("ore", bank_resources[4])?;
    }
    dict.set_item("bank", bank_dict)?;
    
    // Add robber information
    dict.set_item("robber_tile", state.get_robber_tile())?;
    
    // Add player data
    let players_dict = PyDict::new(py);
    // Assuming 4 players (0-3) for now, as this is the standard setup
    // We can't access the num_players directly and get_num_players is private
    for color in 0..4 {
        let player_dict = PyDict::new(py);
        
        // Resources
        let resources = state.get_player_hand(color);
        let resources_dict = PyDict::new(py);
        if resources.len() >= 5 {
            resources_dict.set_item("wood", resources[0])?;
            resources_dict.set_item("brick", resources[1])?;
            resources_dict.set_item("sheep", resources[2])?;
            resources_dict.set_item("wheat", resources[3])?;
            resources_dict.set_item("ore", resources[4])?;
        }
        player_dict.set_item("resources", resources_dict)?;
        
        // Development cards
        let dev_cards = state.get_player_devhand(color);
        let dev_cards_dict = PyDict::new(py);
        if dev_cards.len() >= 5 {
            dev_cards_dict.set_item("knight", dev_cards[0])?;
            dev_cards_dict.set_item("year_of_plenty", dev_cards[1])?;
            dev_cards_dict.set_item("monopoly", dev_cards[2])?;
            dev_cards_dict.set_item("road_building", dev_cards[3])?;
            dev_cards_dict.set_item("victory_point", dev_cards[4])?;
        }
        player_dict.set_item("development_cards", dev_cards_dict)?;
        
        // Victory points
        player_dict.set_item("victory_points", state.get_actual_victory_points(color))?;
        
        // Buildings
        let settlements = state.get_settlements(color);
        let cities = state.get_cities(color);
        let settlements_list = PyList::empty(py);
        let cities_list = PyList::empty(py);
        
        for settlement in settlements {
            if let crate::state::Building::Settlement(_, node_id) = settlement {
                settlements_list.append(node_id)?;
            }
        }
        
        for city in cities {
            if let crate::state::Building::City(_, node_id) = city {
                cities_list.append(node_id)?;
            }
        }
        
        player_dict.set_item("settlements", settlements_list)?;
        player_dict.set_item("cities", cities_list)?;
        
        // Add to players dictionary
        players_dict.set_item(color_to_string(color), player_dict)?;
    }
    
    dict.set_item("players", players_dict)?;
    
    // Add board information
    let board_dict = PyDict::new(py);
    
    // Add buildable locations for current player
    let buildable_nodes = if state.is_initial_build_phase() {
        // In initial build phase, get buildable node IDs for all players
        let mut all_buildable = Vec::new();
        for color in 0..4 {
            let buildable = state.buildable_node_ids(color);
            all_buildable.extend(buildable);
        }
        all_buildable
    } else {
        // In regular gameplay, only show buildable nodes for current player
        state.buildable_node_ids(state.get_current_color())
    };
    
    let buildable_nodes_list = PyList::empty(py);
    for node in buildable_nodes {
        buildable_nodes_list.append(node)?;
    }
    board_dict.set_item("buildable_nodes", buildable_nodes_list)?;
    
    // Add buildable roads for current player
    let buildable_edges = if state.is_initial_build_phase() {
        // In initial phase, get initial road possibilities
        state.initial_road_possibilities(state.get_current_color())
            .iter()
            .filter_map(|action| {
                if let crate::enums::Action::BuildRoad { edge_id, .. } = action {
                    Some(*edge_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    } else {
        // In regular gameplay, only show buildable edges for current player
        state.board_buildable_edges(state.get_current_color())
    };
    
    let buildable_edges_list = PyList::empty(py);
    for edge in buildable_edges {
        let edge_tuple = PyTuple::new(py, [edge.0, edge.1]);
        buildable_edges_list.append(edge_tuple)?;
    }
    board_dict.set_item("buildable_edges", buildable_edges_list)?;
    
    dict.set_item("board", board_dict)?;
    
    Ok(dict.into())
} 