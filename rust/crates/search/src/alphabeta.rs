use std::time::{Duration, Instant};

use catanatron_core::{
    actual_victory_points, apply_checked_with_context, apply_outcome_checked_with_context,
    enumerate_outcomes, generate_actions_with_context, Action, GameContext, Phase, PlayerId,
    Position, Resource, Status, WeightedOutcome,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlphaBetaStats {
    pub completed_depth: u8,
    pub attempted_depth: u8,
    pub nodes: u64,
    pub chance_children: u64,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaBetaResult {
    pub action: Option<Action>,
    pub value: f64,
    pub stats: AlphaBetaStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlphaBetaMode {
    Full,
    SameTurn,
}

pub fn iterative_alpha_beta(
    context: &GameContext,
    root: &Position,
    max_depth: u8,
    budget: Duration,
    mode: AlphaBetaMode,
) -> AlphaBetaResult {
    iterative_alpha_beta_with_evaluator(context, root, max_depth, budget, mode, heuristic)
}

pub fn iterative_alpha_beta_with_evaluator(
    context: &GameContext,
    root: &Position,
    max_depth: u8,
    budget: Duration,
    mode: AlphaBetaMode,
    evaluator: fn(&GameContext, &Position, PlayerId) -> f64,
) -> AlphaBetaResult {
    let started = Instant::now();
    let deadline = started + budget;
    let root_player = root.actor;
    let mut result = AlphaBetaResult {
        action: None,
        value: evaluator(context, root, root_player),
        stats: AlphaBetaStats::default(),
    };
    for depth in 1..=max_depth {
        result.stats.attempted_depth = depth;
        let mut scratch = SearchScratch::default();
        let searched = search(
            context,
            root,
            root_player,
            depth,
            f64::NEG_INFINITY,
            f64::INFINITY,
            deadline,
            mode,
            evaluator,
            &mut scratch,
        );
        result.stats.nodes += scratch.nodes;
        result.stats.chance_children += scratch.chance_children;
        if !searched.complete {
            break;
        }
        result.action = searched.action;
        result.value = searched.value;
        result.stats.completed_depth = depth;
    }
    result.stats.elapsed = started.elapsed();
    result
}

#[derive(Default)]
struct SearchScratch {
    actions: Vec<Action>,
    outcomes: Vec<WeightedOutcome>,
    nodes: u64,
    chance_children: u64,
}

#[derive(Clone, Copy)]
struct NodeResult {
    action: Option<Action>,
    value: f64,
    complete: bool,
}

#[allow(clippy::too_many_arguments)]
fn search(
    context: &GameContext,
    position: &Position,
    root_player: PlayerId,
    depth: u8,
    mut alpha: f64,
    mut beta: f64,
    deadline: Instant,
    mode: AlphaBetaMode,
    evaluator: fn(&GameContext, &Position, PlayerId) -> f64,
    scratch: &mut SearchScratch,
) -> NodeResult {
    scratch.nodes += 1;
    if Instant::now() >= deadline {
        return NodeResult {
            action: None,
            value: evaluator(context, position, root_player),
            complete: false,
        };
    }
    if depth == 0
        || matches!(position.phase, Phase::Terminal)
        || (mode == AlphaBetaMode::SameTurn && position.actor != root_player)
    {
        return NodeResult {
            action: None,
            value: evaluator(context, position, root_player),
            complete: true,
        };
    }
    let mut actions = std::mem::take(&mut scratch.actions);
    generate_actions_with_context(position, context, &mut actions);
    if actions.is_empty() {
        scratch.actions = actions;
        return NodeResult {
            action: None,
            value: evaluator(context, position, root_player),
            complete: true,
        };
    }
    let maximizing = position.actor == root_player;
    let mut best_action = None;
    let mut best_value = if maximizing {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for &action in &actions {
        if Instant::now() >= deadline {
            scratch.actions = actions;
            return NodeResult {
                action: best_action,
                value: best_value,
                complete: false,
            };
        }
        let children = action_children(context, position, action, scratch);
        let mut expected = 0.0;
        for (child, weight, total) in children {
            scratch.chance_children += u64::from(weight != total);
            let searched = search(
                context,
                &child,
                root_player,
                depth - 1,
                alpha,
                beta,
                deadline,
                mode,
                evaluator,
                scratch,
            );
            if !searched.complete {
                scratch.actions = actions;
                return NodeResult {
                    action: best_action,
                    value: best_value,
                    complete: false,
                };
            }
            expected += searched.value * f64::from(weight) / f64::from(total);
        }
        if (maximizing && expected > best_value) || (!maximizing && expected < best_value) {
            best_value = expected;
            best_action = Some(action);
        }
        if maximizing {
            alpha = alpha.max(best_value);
        } else {
            beta = beta.min(best_value);
        }
        if alpha >= beta {
            break;
        }
    }
    scratch.actions = actions;
    NodeResult {
        action: best_action,
        value: best_value,
        complete: true,
    }
}

fn action_children(
    context: &GameContext,
    position: &Position,
    action: Action,
    scratch: &mut SearchScratch,
) -> Vec<(Position, u16, u16)> {
    let mut child = *position;
    let actor = child.actor;
    let transition = apply_checked_with_context(&mut child, context, actor, action)
        .expect("generated action must apply");
    if transition.status != Status::Chance {
        return vec![(child, 1, 1)];
    }
    let mut outcomes = std::mem::take(&mut scratch.outcomes);
    let total = enumerate_outcomes(&child, &mut outcomes);
    let children = outcomes
        .iter()
        .map(|entry| {
            let mut outcome_child = child;
            apply_outcome_checked_with_context(&mut outcome_child, context, entry.outcome)
                .expect("enumerated outcome must apply");
            (outcome_child, entry.weight, total)
        })
        .collect();
    scratch.outcomes = outcomes;
    children
}

pub fn heuristic(context: &GameContext, position: &Position, player: PlayerId) -> f64 {
    let victory = f64::from(actual_victory_points(position, player));
    let mut production = [0.0; 4];
    for raw in 0..position.map.land_tile_count() as u8 {
        let tile = catanatron_core::TileId::new(raw).expect("active tile");
        let assignment = context.layout.tile(tile);
        let Some(number) = assignment.number else {
            continue;
        };
        let probability = f64::from(6_i16 - (7_i16 - i16::from(number)).abs()) / 36.0;
        for node in catanatron_core::land_tile_nodes_on(position.map, tile).expect("active tile") {
            if let Some(owner) =
                catanatron_core::building_owner(position.buildings[usize::from(node.get())])
            {
                production[usize::from(owner.get())] += probability
                    * f64::from(catanatron_core::building_production(
                        position.buildings[usize::from(node.get())],
                    ));
            }
        }
    }
    let opponents = (0..position.player_count)
        .filter(|&seat| seat != player.get())
        .map(|seat| production[usize::from(seat)])
        .sum::<f64>();
    let hand = position.players[usize::from(player.get())]
        .hand
        .iter()
        .map(|&count| f64::from(count))
        .sum::<f64>();
    victory * 3.0e14 + production[usize::from(player.get())] * 1.0e8
        - opponents * 1.0e8 / f64::from(position.player_count - 1)
        + hand
        + f64::from(position.longest_road_lengths[usize::from(player.get())]) * 10.0
        + Resource::ALL
            .iter()
            .filter(|resource| {
                position.players[usize::from(player.get())].hand[resource.index()] > 0
            })
            .count() as f64
            * 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_base, NumberPlacement};

    #[test]
    fn iterative_search_completes_shallow_depth_and_preserves_root() {
        let (context, root) = initialize_base(2, NumberPlacement::OfficialSpiral, 7, 0).unwrap();
        let before = root;
        let result = iterative_alpha_beta(
            &context,
            &root,
            2,
            Duration::from_secs(2),
            AlphaBetaMode::Full,
        );
        assert!(result.action.is_some());
        assert_eq!(result.stats.completed_depth, 2);
        assert!(result.stats.nodes > 1);
        assert_eq!(root, before);
    }
}
