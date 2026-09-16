use std::time::Duration;

use catanatron_core::{
    actual_victory_points, apply_checked_with_context, apply_outcome_checked_with_context,
    enumerate_outcomes, generate_actions_with_context, Action, GameContext, Phase, Position,
    RandomSource, Status,
};

use crate::{
    choose_action, flat_monte_carlo, heuristic, iterative_alpha_beta, rollout, sample_outcome,
    AlphaBetaMode, Policy, RolloutLimits, RolloutScratch, SearchRng,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentKind {
    Simple,
    Random,
    WeightedRandom,
    VictoryPoint,
    ValueFunction,
    GreedyPlayouts,
    AlphaBeta,
    SameTurnAlphaBeta,
    Mcts,
}

#[derive(Clone, Copy, Debug)]
pub struct AgentConfig {
    pub kind: AgentKind,
    pub simulations: u32,
    pub max_depth: u8,
    pub budget: Duration,
}

impl AgentConfig {
    pub const fn new(kind: AgentKind) -> Self {
        Self {
            kind,
            simulations: 25,
            max_depth: 32,
            budget: Duration::from_millis(20),
        }
    }
}

pub fn select_agent_action(
    context: &GameContext,
    position: &Position,
    config: AgentConfig,
    seed: u64,
    actions: &mut Vec<Action>,
) -> Option<Action> {
    generate_actions_with_context(position, context, actions);
    if actions.len() <= 1 {
        return actions.first().copied();
    }
    let mut random = SearchRng::from_seed(crate::derive_seed(
        seed,
        position.turns.into(),
        0,
        crate::StreamKind::Policy,
    ));
    match config.kind {
        AgentKind::Simple => actions.first().copied(),
        AgentKind::Random => choose_action(actions, Policy::Random, &mut random),
        AgentKind::WeightedRandom => choose_action(actions, Policy::Weighted, &mut random),
        AgentKind::VictoryPoint => one_ply_choice(
            context,
            position,
            actions,
            &mut random,
            |child| f64::from(actual_victory_points(child, position.actor)),
            true,
        ),
        AgentKind::ValueFunction => one_ply_choice(
            context,
            position,
            actions,
            &mut random,
            |child| heuristic(context, child, position.actor),
            false,
        ),
        AgentKind::GreedyPlayouts => flat_monte_carlo(
            context,
            position,
            actions,
            config.simulations.saturating_mul(actions.len() as u32),
            seed,
            RolloutLimits::default(),
        )
        .map(|result| result.action),
        AgentKind::AlphaBeta | AgentKind::SameTurnAlphaBeta => iterative_alpha_beta(
            context,
            position,
            config.max_depth,
            config.budget,
            if config.kind == AgentKind::AlphaBeta {
                AlphaBetaMode::Full
            } else {
                AlphaBetaMode::SameTurn
            },
        )
        .action
        .or_else(|| actions.first().copied()),
        AgentKind::Mcts => root_mcts(context, position, actions, config.simulations, seed),
    }
}

fn one_ply_choice(
    context: &GameContext,
    position: &Position,
    actions: &[Action],
    random: &mut impl RandomSource,
    value: impl Fn(&Position) -> f64,
    random_ties: bool,
) -> Option<Action> {
    let mut best = f64::NEG_INFINITY;
    let mut choices = Vec::new();
    for &action in actions {
        let expected = expected_after_action(context, position, action, &value);
        if expected > best {
            best = expected;
            choices.clear();
            choices.push(action);
        } else if expected == best {
            choices.push(action);
        }
    }
    if random_ties {
        choose_action(&choices, Policy::Random, random)
    } else {
        choices.first().copied()
    }
}

fn expected_after_action(
    context: &GameContext,
    position: &Position,
    action: Action,
    value: &impl Fn(&Position) -> f64,
) -> f64 {
    let mut child = *position;
    let actor = child.actor;
    let transition =
        apply_checked_with_context(&mut child, context, actor, action).expect("generated action");
    if transition.status != Status::Chance {
        return value(&child);
    }
    let mut outcomes = Vec::with_capacity(36);
    let total = enumerate_outcomes(&child, &mut outcomes);
    outcomes
        .into_iter()
        .map(|entry| {
            let mut next = child;
            apply_outcome_checked_with_context(&mut next, context, entry.outcome)
                .expect("enumerated outcome");
            value(&next) * f64::from(entry.weight) / f64::from(total)
        })
        .sum()
}

fn root_mcts(
    context: &GameContext,
    position: &Position,
    actions: &[Action],
    simulations: u32,
    seed: u64,
) -> Option<Action> {
    let mut visits = vec![0_u32; actions.len()];
    let mut wins = vec![0.0_f64; actions.len()];
    let mut chance =
        SearchRng::from_seed(crate::derive_seed(seed, 0, 0, crate::StreamKind::Chance));
    let mut outcomes = Vec::with_capacity(36);
    let mut scratch = RolloutScratch::default();
    for simulation in 0..simulations.max(actions.len() as u32) {
        let total = f64::from(simulation.max(1));
        let index = (0..actions.len())
            .max_by(|&left, &right| {
                let score = |i: usize| {
                    if visits[i] == 0 {
                        f64::INFINITY
                    } else {
                        wins[i] / f64::from(visits[i])
                            + (2.0 * total.ln() / f64::from(visits[i])).sqrt()
                    }
                };
                score(left).total_cmp(&score(right))
            })
            .expect("non-empty menu");
        let mut child = *position;
        let actor = child.actor;
        let transition = apply_checked_with_context(&mut child, context, actor, actions[index])
            .expect("generated action");
        if transition.status == Status::Chance {
            let outcome =
                sample_outcome(&child, &mut chance, &mut outcomes).expect("chance outcome");
            apply_outcome_checked_with_context(&mut child, context, outcome)
                .expect("sampled outcome");
        }
        let result = rollout(
            context,
            &child,
            Policy::Random,
            seed.wrapping_add(u64::from(simulation)),
            RolloutLimits::default(),
            &mut scratch,
        );
        visits[index] += 1;
        wins[index] += f64::from(result.winner == Some(position.actor));
    }
    (0..actions.len())
        .max_by(|&left, &right| {
            (wins[left] / f64::from(visits[left]))
                .total_cmp(&(wins[right] / f64::from(visits[right])))
        })
        .map(|index| actions[index])
}

pub fn play_agents(
    context: &GameContext,
    root: &Position,
    agents: &[AgentConfig],
    seed: u64,
    limits: RolloutLimits,
) -> crate::RolloutResult {
    assert_eq!(agents.len(), usize::from(root.player_count));
    let mut position = *root;
    let mut chance =
        SearchRng::from_seed(crate::derive_seed(seed, 0, 0, crate::StreamKind::Chance));
    let mut outcomes = Vec::with_capacity(36);
    let mut actions = Vec::with_capacity(256);
    let mut player_actions = 0;
    loop {
        let truncation = if position.turns >= limits.turn_limit {
            Some(catanatron_core::Truncation::TurnLimit)
        } else if player_actions >= limits.action_limit {
            Some(catanatron_core::Truncation::ActionLimit)
        } else {
            None
        };
        if let Some(reason) = truncation {
            return crate::RolloutResult {
                winner: None,
                truncation: Some(reason),
                turns: position.turns,
                player_actions,
            };
        }
        let transition = if matches!(position.phase, Phase::Chance { .. }) {
            let outcome = sample_outcome(&position, &mut chance, &mut outcomes).expect("chance");
            apply_outcome_checked_with_context(&mut position, context, outcome)
                .expect("sampled outcome")
        } else {
            let actor = position.actor;
            let action = select_agent_action(
                context,
                &position,
                agents[usize::from(actor.get())],
                seed.wrapping_add(u64::from(player_actions)),
                &mut actions,
            )
            .expect("legal action");
            player_actions += 1;
            apply_checked_with_context(&mut position, context, actor, action)
                .expect("selected action")
        };
        match transition.status {
            Status::Won(winner) => {
                return crate::RolloutResult {
                    winner: Some(winner),
                    truncation: None,
                    turns: position.turns,
                    player_actions,
                }
            }
            Status::Truncated(reason) => {
                return crate::RolloutResult {
                    winner: None,
                    truncation: Some(reason),
                    turns: position.turns,
                    player_actions,
                }
            }
            Status::Decision | Status::Chance => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_base, NumberPlacement};

    #[test]
    fn reconstructed_agents_choose_legal_actions_without_mutating_root() {
        let (context, root) = initialize_base(2, NumberPlacement::OfficialSpiral, 3, 0).unwrap();
        for kind in [
            AgentKind::Simple,
            AgentKind::Random,
            AgentKind::WeightedRandom,
            AgentKind::VictoryPoint,
            AgentKind::ValueFunction,
            AgentKind::GreedyPlayouts,
            AgentKind::AlphaBeta,
            AgentKind::SameTurnAlphaBeta,
            AgentKind::Mcts,
        ] {
            let mut actions = Vec::new();
            let action = select_agent_action(
                &context,
                &root,
                AgentConfig {
                    kind,
                    simulations: 2,
                    max_depth: 1,
                    budget: Duration::from_millis(20),
                },
                7,
                &mut actions,
            )
            .unwrap();
            assert!(actions.contains(&action), "{kind:?}");
        }
    }
}
