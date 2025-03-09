use log::{debug, info};
use std::collections::HashMap;
use std::rc::Rc;

use crate::enums::{Action, GameConfiguration, MapType};
use crate::global_state::GlobalState;
use crate::map_instance::MapInstance;
use crate::players::{Player, RandomPlayer};
use crate::state::State;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub fn play_game(
    global_state: GlobalState,
    config: GameConfiguration,
    players: HashMap<u8, Box<dyn Player>>,
) -> Option<u8> {
    debug!(
        "play_game: config={:?}, players={:?}",
        config,
        players.keys().collect::<Vec<_>>()
    );

    let map_instance = MapInstance::new(
        &global_state.base_map_template,
        &global_state.dice_probas,
        0,
    );
    let rc_config = Rc::new(config);
    info!("Playing game with configuration: {:?}", rc_config);
    let mut state = State::new(rc_config.clone(), Rc::new(map_instance));

    debug!(
        "State initialized: current_tick_seat={}, current_color={}",
        state.get_current_tick_seat(),
        state.get_current_color()
    );

    info!("Seat order: {:?}", state.get_seating_order());
    let mut num_ticks = 0;
    while state.winner().is_none() && num_ticks < rc_config.max_ticks {
        debug!("Tick {:?} =====", num_ticks);
        play_tick(&players, &mut state);
        num_ticks += 1;
    }
    state.winner()
}

fn play_tick(players: &HashMap<u8, Box<dyn Player>>, state: &mut State) -> Action {
    let current_color = state.get_current_color();
    debug!(
        "play_tick: current_color={}, players={:?}, action_prompt={:?}",
        current_color,
        players.keys().collect::<Vec<_>>(),
        state.get_action_prompt()
    );

    let current_player = match players.get(&current_color) {
        Some(player) => player,
        None => {
            debug!(
                "ERROR: No player found for color {}. Available players: {:?}",
                current_color,
                players.keys().collect::<Vec<_>>()
            );
            panic!("No player found for color {}", current_color);
        }
    };

    let playable_actions = state.generate_playable_actions();
    debug!(
        "Player {:?} has {:?} playable actions",
        current_color, playable_actions
    );
    let action = current_player.decide(state, &playable_actions);
    debug!(
        "Player {:?} decided to play action {:?}",
        current_color, action
    );

    state.apply_action(action);
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        enums::{Action, ActionPrompt, MapType},
        players::RandomPlayer,
    };

    fn setup_game(
        num_players: u8,
    ) -> (GlobalState, GameConfiguration, HashMap<u8, Box<dyn Player>>) {
        let global_state = GlobalState::new();
        let config = GameConfiguration {
            discard_limit: 7,
            vps_to_win: 10,
            map_type: MapType::Base,
            num_players,
            max_ticks: 8, // TODO: Change!
        };
        let mut players: HashMap<u8, Box<dyn Player>> = HashMap::new();
        players.insert(0, Box::new(RandomPlayer {}));
        players.insert(1, Box::new(RandomPlayer {}));
        players.insert(2, Box::new(RandomPlayer {}));
        players.insert(3, Box::new(RandomPlayer {}));
        (global_state, config, players)
    }

    #[test]
    fn test_game_creation() {
        let (global_state, config, players) = setup_game(4);

        let result = play_game(global_state, config, players);
        assert_eq!(result, None);
    }

    #[test]
    fn test_initial_build_phase_four_player() {
        let (global_state, config, players) = setup_game(4);
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );
        let rc_config = Rc::new(config);
        let mut state = State::new(rc_config.clone(), Rc::new(map_instance));

        let seating_order = state.get_seating_order();
        let first_player = seating_order[0];
        let second_player = seating_order[1];
        let third_player = seating_order[2];
        let fourth_player = seating_order[3];

        // first player settlement
        let playable_actions = state.generate_playable_actions();
        assert_eq!(playable_actions.len(), 54);
        assert_all_build_settlements(playable_actions, first_player);
        play_tick(&players, &mut state);

        // first player road
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 2);
        assert_all_build_roads(playable_actions, first_player);
        assert!(state.is_initial_build_phase());
        play_tick(&players, &mut state);

        // second player settlement: assert at 50-51 actions and all are build settlement
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 50 && playable_actions.len() <= 51);
        assert_all_build_settlements(playable_actions, second_player);
        play_tick(&players, &mut state);

        // second player road: assert at least 2 actions and all are build road
        let playable_actions = state.generate_playable_actions();
        assert!(playable_actions.len() >= 2);
        assert_all_build_roads(playable_actions, second_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // third player settlement

        // third player road
        let playable_actions = state.generate_playable_actions();
        assert_all_build_roads(playable_actions, third_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // fourth player settlement
        play_tick(&players, &mut state); // fourth player road

        // fourth player settlement 2
        assert!(state.is_initial_build_phase());
        let playable_actions = state.generate_playable_actions();
        assert_all_build_settlements(playable_actions, fourth_player);
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // fourth player road
        play_tick(&players, &mut state); // third player settlement 2
        play_tick(&players, &mut state); // third player road
        let second_player_second_settlement_action = play_tick(&players, &mut state);
        let second_player_second_node_id;
        if let Action::BuildSettlement {
            color: player,
            node_id,
        } = second_player_second_settlement_action
        {
            assert_eq!(player, second_player);
            second_player_second_node_id = node_id;
        } else {
            panic!("Expected Action::BuildSettlement");
        }
        debug!("{}", second_player_second_node_id);

        // second player road 2
        let playable_actions = state.generate_playable_actions();
        assert_all_build_roads(playable_actions.clone(), second_player);
        // assert playable_actions are connected to the last settlement
        assert!(playable_actions.iter().all(|e| {
            if let Action::BuildRoad { edge_id, .. } = e {
                second_player_second_node_id == edge_id.0
                    || second_player_second_node_id == edge_id.1
            } else {
                false
            }
        }));
        play_tick(&players, &mut state);

        play_tick(&players, &mut state); // first player settlement 2
        play_tick(&players, &mut state); // first player road

        // Assert that the initial build phase is over and its the first player's turn
        assert!(!state.is_initial_build_phase());
        assert_eq!(state.get_current_color(), first_player);
        assert!(matches!(state.get_action_prompt(), ActionPrompt::PlayTurn));

        // TODO: Assert players have money of their second house
    }

    #[test]
    fn test_initial_build_phase_two_player() {
        let (global_state, config, players) = setup_game(2);
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0,
        );
        let rc_config = Rc::new(config);
        let mut state = State::new(rc_config.clone(), Rc::new(map_instance));

        let seating_order = state.get_seating_order();
        let first_player = seating_order[0];

        for _ in 0..8 {
            assert!(state.is_initial_build_phase());
            play_tick(&players, &mut state);
        }

        // Assert that the initial build phase is over and its the first player's turn
        assert!(!state.is_initial_build_phase());
        assert_eq!(state.get_current_color(), first_player);
        assert!(matches!(state.get_action_prompt(), ActionPrompt::PlayTurn));
    }

    fn assert_all_build_settlements(playable_actions: Vec<Action>, player: u8) {
        assert!(
            playable_actions.iter().all(|e| {
                if let Action::BuildSettlement { color, .. } = e {
                    *color == player
                } else {
                    false
                }
            }),
            "Expected all actions to be BuildSettlement for player {:?}",
            player
        );
    }

    fn assert_all_build_roads(playable_actions: Vec<Action>, player: u8) {
        assert!(
            playable_actions.iter().all(|e| {
                if let Action::BuildRoad { color, .. } = e {
                    *color == player
                } else {
                    false
                }
            }),
            "Expected all actions to be BuildRoad for player {:?}",
            player
        );
    }
}

#[pyclass(unsendable)]
pub struct Game {
    num_players: usize,
    config: GameConfiguration,
    winner: Option<u8>,
    state: Option<State>,
    players: HashMap<u8, Box<dyn Player>>,
    accumulators: Vec<PyObject>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (
        players,
        _seed = None,
        discard_limit = 7,
        vps_to_win = 10,
        _map_instance = None,
        _initialize = true
    ))]
    fn new(
        py: Python,
        players: &PyList,
        _seed: Option<i64>,
        discard_limit: u8,
        vps_to_win: u8,
        _map_instance: Option<&PyAny>,  // Unused, prefix with underscore
        _initialize: bool,              // Unused, prefix with underscore
    ) -> PyResult<Self> {
        let num_players = players.len();
        let config = GameConfiguration {
            discard_limit,
            vps_to_win,
            map_type: MapType::Base, // For now, hardcoded to Base
            num_players: num_players as u8,
            max_ticks: 10000,
        };

        let mut player_map = HashMap::new();

        // Check if we're integrating with Python players
        let use_python_players = true;  // Set to true to enable Python player integration
        
        if use_python_players {
            // Create PythonPlayerWrappers for each Python player
            for (i, player) in players.iter().enumerate() {
                let python_player = player.to_object(py);
                player_map.insert(
                    i as u8, 
                    Box::new(crate::players::PythonPlayerWrapper::new(python_player)) as Box<dyn Player>
                );
            }
        } else {
            // Use RandomPlayers for each color
            for i in 0..num_players {
                player_map.insert(i as u8, Box::new(RandomPlayer {}) as Box<dyn Player>);
            }
        }

        Ok(Game {
            num_players,
            config,
            winner: None,
            state: None,
            players: player_map,
            accumulators: Vec::new(),
        })
    }

    fn play(&mut self, py: Python, accumulators: Option<Vec<PyObject>>, decide_fn: Option<PyObject>) -> PyResult<Option<u8>> {
        // If accumulators are provided, store them
        if let Some(accs) = accumulators {
            self.accumulators = accs;
        }

        // Call any accumulators' before method
        for accumulator in &self.accumulators {
            let _ = accumulator.call_method1(py, "before", (Option::<PyObject>::None.to_object(py),));
        }

        // Initialize and play the game
        let global_state = GlobalState::new();
        
        // Create a new state - in the future, we'll need to support custom map instances
        let map_instance = MapInstance::new(
            &global_state.base_map_template,
            &global_state.dice_probas,
            0, // No seed for now
        );
        let rc_config = Rc::new(self.config.clone());
        let state = State::new(rc_config.clone(), Rc::new(map_instance));
        self.state = Some(state);

        // Play the game until a winner or max ticks
        let mut num_ticks = 0;
        while self.state.as_ref().unwrap().winner().is_none() && num_ticks < rc_config.max_ticks {
            self.play_tick(py, decide_fn.clone())?;
            num_ticks += 1;
        }

        // Store the winner and call any accumulators' after method
        self.winner = self.state.as_ref().unwrap().winner();
        
        for accumulator in &self.accumulators {
            let _ = accumulator.call_method1(py, "after", (Option::<PyObject>::None.to_object(py),));
        }

        match self.winner {
            Some(winner) => info!("Game completed - Player {} won!", winner),
            None => info!("Game ended without a winner"),
        }

        Ok(self.winner)
    }

    fn play_tick(&mut self, py: Python, _decide_fn: Option<PyObject>) -> PyResult<PyObject> {
        if self.state.is_none() {
            // Initialize state if not already done
            let global_state = GlobalState::new();
            let map_instance = MapInstance::new(
                &global_state.base_map_template,
                &global_state.dice_probas,
                0,
            );
            let rc_config = Rc::new(self.config.clone());
            self.state = Some(State::new(rc_config.clone(), Rc::new(map_instance)));
        }
        
        let state = self.state.as_mut().unwrap();
        let current_color = state.get_current_color();
        
        let playable_actions = state.generate_playable_actions();
        debug!(
            "Player {:?} has {:?} playable actions",
            current_color, playable_actions
        );
        
        // Get the current player
        let current_player = match self.players.get(&current_color) {
            Some(player) => player,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("No player found for color {}", current_color)
                ));
            }
        };
        
        // Let the player decide what to do
        let action = current_player.decide(state, &playable_actions);
        debug!(
            "Player {:?} decided to play action {:?}",
            current_color, action
        );
        
        // Call accumulators' step method before applying the action
        for accumulator in &self.accumulators {
            let _ = accumulator.call_method1(py, "step", (Option::<PyObject>::None.to_object(py), format!("{:?}", action).to_object(py)));
        }
        
        // Apply the action
        state.apply_action(action);
        
        // Return the action as a Python object
        Ok(format!("{:?}", action).to_object(py))
    }

    fn get_num_players(&self) -> usize {
        self.num_players
    }

    fn get_winner(&self) -> Option<u8> {
        self.winner
    }
    
    // New methods to match Python API
    fn winning_color(&self) -> Option<u8> {
        self.winner
    }
    
    // State inspection methods - placeholders for now
    fn get_state_repr(&self, py: Python) -> PyResult<PyObject> {
        if let Some(state) = &self.state {
            Ok(format!("{:?}", state).to_object(py))
        } else {
            Ok(Option::<PyObject>::None.to_object(py))
        }
    }
}
