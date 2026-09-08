#![forbid(unsafe_code)]

use catanatron_core::{
    apply_checked_with_context, apply_outcome_checked_with_context, generate_actions_with_context,
    Action, GameContext, Phase, Position, Status,
};
use catanatron_search::{
    derive_seed, initialize_base, initialize_mini, initialize_tournament, rollout, sample_outcome,
    NumberPlacement, Policy, RolloutLimits, RolloutResult, RolloutScratch, SearchRng, StreamKind,
};
use numpy::{ndarray::Array2, IntoPyArray};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

const OBSERVATION_SCHEMA_VERSION: u16 = 1;
const ACTION_SCHEMA_VERSION: u16 = 1;
const FEATURE_COUNT: usize = 1 + 1 + 1 + 5 + 54 + 72 + 4 * (5 + 5 + 3 + 2);

#[derive(Clone, Copy)]
enum BoardConfig {
    Base,
    Mini,
    Tournament,
}

impl BoardConfig {
    fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "BASE" => Ok(Self::Base),
            "MINI" => Ok(Self::Mini),
            "TOURNAMENT" => Ok(Self::Tournament),
            _ => Err(PyValueError::new_err(
                "map must be BASE, MINI, or TOURNAMENT",
            )),
        }
    }
}

#[pyclass]
struct Batch {
    contexts: Vec<GameContext>,
    positions: Vec<Position>,
    generations: Vec<u32>,
    chance_counters: Vec<u64>,
    chance_seeds: Vec<u64>,
    players: u8,
    board: BoardConfig,
}

#[pymethods]
impl Batch {
    #[new]
    #[pyo3(signature = (size, players=4, map="BASE", seed=0))]
    fn new(size: usize, players: u8, map: &str, seed: u64) -> PyResult<Self> {
        if size == 0 {
            return Err(PyValueError::new_err("size must be positive"));
        }
        let board = BoardConfig::parse(map)?;
        let mut contexts = Vec::with_capacity(size);
        let mut positions = Vec::with_capacity(size);
        for index in 0..size {
            let (context, position) = initialize(board, players, seed, index as u64)?;
            contexts.push(context);
            positions.push(position);
        }
        Ok(Self {
            contexts,
            positions,
            generations: vec![1; size],
            chance_counters: vec![0; size],
            chance_seeds: (0..size).map(|index| seed ^ index as u64).collect(),
            players,
            board,
        })
    }

    fn reset_many<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        seeds: Vec<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        validate_indices(&indices, self.positions.len())?;
        if indices.len() != seeds.len() {
            return Err(PyValueError::new_err(
                "indices and seeds must have equal length",
            ));
        }
        let replacements = py.detach(|| {
            indices
                .iter()
                .zip(&seeds)
                .map(|(&index, &seed)| initialize(self.board, self.players, seed, index as u64))
                .collect::<PyResult<Vec<_>>>()
        })?;
        for ((&index, (context, position)), &seed) in indices.iter().zip(replacements).zip(&seeds) {
            self.contexts[index] = context;
            self.positions[index] = position;
            self.generations[index] = next_generation(self.generations[index])?;
            self.chance_counters[index] = 0;
            self.chance_seeds[index] = seed;
        }
        observations(
            py,
            &self.contexts,
            &self.positions,
            &self.generations,
            &indices,
        )
    }

    fn observe_many<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        validate_indices(&indices, self.positions.len())?;
        observations(
            py,
            &self.contexts,
            &self.positions,
            &self.generations,
            &indices,
        )
    }

    fn step_many<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        action_ids: Vec<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        validate_indices(&indices, self.positions.len())?;
        if indices.len() != action_ids.len() {
            return Err(PyValueError::new_err(
                "indices and action_ids must have equal length",
            ));
        }
        let mut actions = Vec::with_capacity(indices.len());
        for (&index, &action_id) in indices.iter().zip(&action_ids) {
            let generation = (action_id >> 32) as u32;
            let row = action_id as u32 as usize;
            if generation != self.generations[index] {
                return Err(PyValueError::new_err("stale dynamic action id"));
            }
            let mut menu = Vec::new();
            generate_actions_with_context(&self.positions[index], &self.contexts[index], &mut menu);
            actions.push(
                *menu
                    .get(row)
                    .ok_or_else(|| PyValueError::new_err("action id row is out of range"))?,
            );
        }

        let results = py
            .detach(|| {
                indices
                    .iter()
                    .zip(actions)
                    .map(|(&index, action)| self.step_one(index, action))
                    .collect::<Result<Vec<_>, String>>()
            })
            .map_err(PyValueError::new_err)?;

        let actors: Vec<u8> = indices
            .iter()
            .map(|&i| self.positions[i].actor.get())
            .collect();
        let terminal: Vec<bool> = results
            .iter()
            .map(|result| matches!(result, Status::Won(_)))
            .collect();
        let truncated = vec![false; indices.len()];
        let mut rewards = Vec::with_capacity(indices.len() * usize::from(self.players));
        for result in &results {
            for player in 0..self.players {
                rewards.push(match result {
                    Status::Won(winner) if winner.get() == player => 1_i8,
                    Status::Won(_) => -1,
                    _ => 0,
                });
            }
        }
        let dict = observations(
            py,
            &self.contexts,
            &self.positions,
            &self.generations,
            &indices,
        )?;
        dict.set_item("actors", actors.into_pyarray(py))?;
        dict.set_item(
            "rewards",
            Array2::from_shape_vec((indices.len(), usize::from(self.players)), rewards)
                .expect("reward shape")
                .into_pyarray(py),
        )?;
        dict.set_item("terminal", terminal.into_pyarray(py))?;
        dict.set_item("truncated", truncated.into_pyarray(py))?;
        Ok(dict)
    }

    #[pyo3(signature = (indices, seeds, turn_limit=1000, action_limit=100000, policy="weighted", threads=1))]
    #[allow(clippy::too_many_arguments)]
    fn rollout_many<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
        seeds: Vec<u64>,
        turn_limit: u16,
        action_limit: u32,
        policy: &str,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        validate_indices(&indices, self.positions.len())?;
        if indices.len() != seeds.len() {
            return Err(PyValueError::new_err(
                "indices and seeds must have equal length",
            ));
        }
        let policy = match policy {
            "random" => Policy::Random,
            "weighted" => Policy::Weighted,
            _ => return Err(PyValueError::new_err("policy must be random or weighted")),
        };
        if indices.is_empty() {
            return rollout_result_dict(py, &[], self.players);
        }
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        let results = py.detach(|| {
            rollout_selected(
                &self.contexts,
                &self.positions,
                &indices,
                &seeds,
                policy,
                RolloutLimits {
                    turn_limit,
                    action_limit,
                },
                threads,
            )
        });
        rollout_result_dict(py, &results, self.players)
    }
}

impl Batch {
    fn step_one(&mut self, index: usize, action: Action) -> Result<Status, String> {
        let position = &mut self.positions[index];
        let actor = position.actor;
        let mut status = apply_checked_with_context(position, &self.contexts[index], actor, action)
            .map_err(|error| format!("{error:?}"))?
            .status;
        let mut outcomes = Vec::with_capacity(36);
        while matches!(position.phase, Phase::Chance { .. }) {
            let counter = self.chance_counters[index];
            let mut rng = SearchRng::from_seed(derive_seed(
                self.chance_seeds[index],
                index as u64,
                counter,
                StreamKind::Chance,
            ));
            let outcome = sample_outcome(position, &mut rng, &mut outcomes)
                .ok_or("chance phase has no outcome")?;
            status = apply_outcome_checked_with_context(position, &self.contexts[index], outcome)
                .map_err(|error| format!("{error:?}"))?
                .status;
            self.chance_counters[index] =
                counter.checked_add(1).ok_or("chance counter exhausted")?;
        }
        self.generations[index] =
            next_generation(self.generations[index]).map_err(|error| error.to_string())?;
        Ok(status)
    }
}

fn initialize(
    board: BoardConfig,
    players: u8,
    seed: u64,
    index: u64,
) -> PyResult<(GameContext, Position)> {
    let result = match board {
        BoardConfig::Base => initialize_base(players, NumberPlacement::OfficialSpiral, seed, index),
        BoardConfig::Mini => initialize_mini(players, NumberPlacement::OfficialSpiral, seed, index),
        BoardConfig::Tournament => initialize_tournament(players),
    };
    result.map_err(|error| PyValueError::new_err(format!("{error:?}")))
}

fn validate_indices(indices: &[usize], size: usize) -> PyResult<()> {
    let mut seen = vec![false; size];
    for &index in indices {
        if index >= size {
            return Err(PyValueError::new_err("environment index is out of range"));
        }
        if seen[index] {
            return Err(PyValueError::new_err("duplicate environment index"));
        }
        seen[index] = true;
    }
    Ok(())
}

fn next_generation(current: u32) -> PyResult<u32> {
    current
        .checked_add(1)
        .ok_or_else(|| PyValueError::new_err("dynamic action generation exhausted"))
}

fn observations<'py>(
    py: Python<'py>,
    contexts: &[GameContext],
    positions: &[Position],
    generations: &[u32],
    indices: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let mut features = Vec::with_capacity(indices.len() * FEATURE_COUNT);
    let mut actors = Vec::with_capacity(indices.len());
    let mut offsets = Vec::with_capacity(indices.len() + 1);
    let mut action_ids = Vec::new();
    offsets.push(0_u32);
    for &index in indices {
        encode_features(&positions[index], &mut features);
        actors.push(positions[index].actor.get());
        let mut menu = Vec::new();
        generate_actions_with_context(&positions[index], &contexts[index], &mut menu);
        action_ids
            .extend((0..menu.len()).map(|row| (u64::from(generations[index]) << 32) | row as u64));
        offsets.push(action_ids.len() as u32);
    }
    let dict = PyDict::new(py);
    dict.set_item("observation_schema_version", OBSERVATION_SCHEMA_VERSION)?;
    dict.set_item("action_schema_version", ACTION_SCHEMA_VERSION)?;
    dict.set_item(
        "features",
        Array2::from_shape_vec((indices.len(), FEATURE_COUNT), features)
            .expect("feature shape")
            .into_pyarray(py),
    )?;
    dict.set_item("actors", actors.into_pyarray(py))?;
    dict.set_item("menu_offsets", offsets.into_pyarray(py))?;
    dict.set_item("action_ids", action_ids.into_pyarray(py))?;
    Ok(dict)
}

fn encode_features(position: &Position, out: &mut Vec<i16>) {
    out.push(i16::from(position.player_count));
    out.push(i16::from(position.actor.get()));
    out.push(i16::from(position.robber));
    out.extend(position.bank.map(i16::from));
    out.extend(position.buildings.map(i16::from));
    out.extend(position.roads.map(i16::from));
    for player in position.players {
        out.extend(player.hand.map(i16::from));
        out.extend(player.dev.map(i16::from));
        out.extend(player.pieces.map(i16::from));
        out.push(i16::from(player.played_knights));
        out.push(i16::from(player.played_dev));
    }
}

fn rollout_result_dict<'py>(
    py: Python<'py>,
    results: &[catanatron_search::RolloutResult],
    players: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let winners: Vec<i8> = results
        .iter()
        .map(|result| result.winner.map_or(-1, |winner| winner.get() as i8))
        .collect();
    let truncated: Vec<bool> = results
        .iter()
        .map(|result| result.truncation.is_some())
        .collect();
    let mut rewards = Vec::with_capacity(results.len() * usize::from(players));
    for result in results {
        for player in 0..players {
            rewards.push(match result.winner {
                Some(winner) if winner.get() == player => 1_i8,
                Some(_) => -1,
                None => 0,
            });
        }
    }
    let dict = PyDict::new(py);
    dict.set_item("winners", winners.into_pyarray(py))?;
    dict.set_item(
        "rewards",
        Array2::from_shape_vec((results.len(), usize::from(players)), rewards)
            .expect("reward shape")
            .into_pyarray(py),
    )?;
    dict.set_item("truncated", truncated.into_pyarray(py))?;
    Ok(dict)
}

fn rollout_selected(
    contexts: &[GameContext],
    positions: &[Position],
    indices: &[usize],
    seeds: &[u64],
    policy: Policy,
    limits: RolloutLimits,
    threads: usize,
) -> Vec<RolloutResult> {
    let workers = threads.min(indices.len());
    let chunk_size = indices.len().div_ceil(workers);
    let chunks = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..indices.len()).step_by(chunk_size) {
            let end = (start + chunk_size).min(indices.len());
            handles.push(scope.spawn(move || {
                let mut scratch = RolloutScratch::default();
                let results = (start..end)
                    .map(|row| {
                        rollout(
                            &contexts[indices[row]],
                            &positions[indices[row]],
                            policy,
                            seeds[row],
                            limits,
                            &mut scratch,
                        )
                    })
                    .collect();
                (start, results)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("rollout worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut results = Vec::with_capacity(indices.len());
    for (_, mut chunk) in chunks {
        results.append(&mut chunk);
    }
    results
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Batch>()?;
    Ok(())
}
