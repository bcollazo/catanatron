use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::enums::Action as RustAction;

/// Python-facing Action class that wraps the Rust Action enum
#[pyclass]
#[derive(Clone)]
pub struct Action {
    #[pyo3(get)]
    pub action_type: String,
    
    #[pyo3(get)]
    pub params: Py<PyDict>,
    
    // Internal Rust action (not exposed to Python)
    pub(crate) rust_action: RustAction,
}

#[pymethods]
impl Action {
    #[staticmethod]
    fn build_settlement(py: Python, color: u8, node_id: u8) -> PyResult<Self> {
        let rust_action = RustAction::BuildSettlement { color, node_id };
        let params = PyDict::new(py);
        params.set_item("color", color)?;
        params.set_item("node_id", node_id)?;
        
        Ok(Self {
            action_type: "BuildSettlement".to_string(),
            params: params.into(),
            rust_action,
        })
    }
    
    #[staticmethod]
    fn build_road(py: Python, color: u8, edge: (u8, u8)) -> PyResult<Self> {
        let rust_action = RustAction::BuildRoad { color, edge_id: edge };
        let params = PyDict::new(py);
        params.set_item("color", color)?;
        params.set_item("edge", edge)?;
        
        Ok(Self {
            action_type: "BuildRoad".to_string(),
            params: params.into(),
            rust_action,
        })
    }
    
    #[staticmethod]
    fn build_city(py: Python, color: u8, node_id: u8) -> PyResult<Self> {
        let rust_action = RustAction::BuildCity { color, node_id };
        let params = PyDict::new(py);
        params.set_item("color", color)?;
        params.set_item("node_id", node_id)?;
        
        Ok(Self {
            action_type: "BuildCity".to_string(),
            params: params.into(),
            rust_action,
        })
    }
    
    #[staticmethod]
    fn end_turn(py: Python, color: u8) -> PyResult<Self> {
        let rust_action = RustAction::EndTurn { color };
        let params = PyDict::new(py);
        params.set_item("color", color)?;
        
        Ok(Self {
            action_type: "EndTurn".to_string(),
            params: params.into(),
            rust_action,
        })
    }
    
    #[staticmethod]
    fn roll(py: Python, color: u8, dice_opt: Option<(u8, u8)>) -> PyResult<Self> {
        let rust_action = RustAction::Roll { color, dice_opt };
        let params = PyDict::new(py);
        params.set_item("color", color)?;
        if let Some(dice) = dice_opt {
            params.set_item("dice", dice)?;
        }
        
        Ok(Self {
            action_type: "Roll".to_string(),
            params: params.into(),
            rust_action,
        })
    }
    
    #[staticmethod]
    fn move_robber(py: Python, color: u8, coordinate: (i8, i8, i8), victim_opt: Option<u8>) -> PyResult<Self> {
        let rust_action = RustAction::MoveRobber { color, coordinate, victim_opt };
        let params = PyDict::new(py);
        params.set_item("color", color)?;
        params.set_item("coordinate", coordinate)?;
        if let Some(victim) = victim_opt {
            params.set_item("victim", victim)?;
        }
        
        Ok(Self {
            action_type: "MoveRobber".to_string(),
            params: params.into(),
            rust_action,
        })
    }
    
    // String representation for debugging
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.rust_action))
    }
    
    // Representation for Python REPL
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
    
    // Equality comparison
    fn __eq__(&self, other: &Self) -> PyResult<bool> {
        Ok(format!("{:?}", self.rust_action) == format!("{:?}", other.rust_action))
    }
}

/// Convert a Rust Action to a Python Action object
pub fn rust_to_py_action(py: Python, action: &RustAction) -> PyResult<Py<Action>> {
    let params = PyDict::new(py);
    
    let (action_type, params_dict) = match action {
        RustAction::BuildSettlement { color, node_id } => {
            params.set_item("color", color)?;
            params.set_item("node_id", node_id)?;
            ("BuildSettlement", params)
        },
        RustAction::BuildRoad { color, edge_id } => {
            params.set_item("color", color)?;
            params.set_item("edge", edge_id)?;
            ("BuildRoad", params)
        },
        RustAction::BuildCity { color, node_id } => {
            params.set_item("color", color)?;
            params.set_item("node_id", node_id)?;
            ("BuildCity", params)
        },
        RustAction::EndTurn { color } => {
            params.set_item("color", color)?;
            ("EndTurn", params)
        },
        RustAction::Roll { color, dice_opt } => {
            params.set_item("color", color)?;
            if let Some(dice) = dice_opt {
                params.set_item("dice", dice)?;
            }
            ("Roll", params)
        },
        RustAction::MoveRobber { color, coordinate, victim_opt } => {
            params.set_item("color", color)?;
            params.set_item("coordinate", coordinate)?;
            if let Some(victim) = victim_opt {
                params.set_item("victim", victim)?;
            }
            ("MoveRobber", params)
        },
        // Add other action types as needed
        _ => {
            params.set_item("action", format!("{:?}", action))?;
            ("Unknown", params)
        }
    };
    
    let py_action = Py::new(py, Action {
        action_type: action_type.to_string(),
        params: params_dict.into(),
        rust_action: *action,
    })?;
    
    Ok(py_action)
}

/// Try to convert a Python object to a Rust Action
pub fn py_to_rust_action(action: &Action) -> RustAction {
    // Simply return the internal Rust action
    action.rust_action
} 