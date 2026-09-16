use catanatron_core::{
    apply_checked_with_context, apply_outcome_checked_with_context, generate_actions_with_context,
    GameContext, Phase, PlayerId, Position, Status, Truncation, WeightedOutcome,
};

use crate::{choose_action, derive_seed, sample_outcome, Policy, SearchRng, StreamKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutLimits {
    pub turn_limit: u16,
    pub action_limit: u32,
}

impl Default for RolloutLimits {
    fn default() -> Self {
        Self {
            turn_limit: 1_000,
            action_limit: 100_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutResult {
    pub winner: Option<PlayerId>,
    pub truncation: Option<Truncation>,
    pub turns: u16,
    pub player_actions: u32,
}

pub struct RolloutScratch {
    actions: Vec<catanatron_core::Action>,
    outcomes: Vec<WeightedOutcome>,
}

impl Default for RolloutScratch {
    fn default() -> Self {
        Self {
            actions: Vec::with_capacity(256),
            outcomes: Vec::with_capacity(36),
        }
    }
}

pub fn rollout(
    context: &GameContext,
    root: &Position,
    policy: Policy,
    seed: u64,
    limits: RolloutLimits,
    scratch: &mut RolloutScratch,
) -> RolloutResult {
    rollout_until(context, root, policy, seed, limits, scratch, || false)
}

pub fn rollout_until(
    context: &GameContext,
    root: &Position,
    policy: Policy,
    seed: u64,
    limits: RolloutLimits,
    scratch: &mut RolloutScratch,
    mut should_stop: impl FnMut() -> bool,
) -> RolloutResult {
    let mut position = *root;
    let mut chance_rng = SearchRng::from_seed(derive_seed(seed, 0, 0, StreamKind::Chance));
    let mut policy_rng = SearchRng::from_seed(derive_seed(seed, 0, 0, StreamKind::Policy));
    let mut player_actions = 0;
    loop {
        debug_assert_conservation(&position);
        if should_stop() {
            return result(
                &position,
                player_actions,
                None,
                Some(Truncation::ActionLimit),
            );
        }
        if position.turns >= limits.turn_limit {
            return result(&position, player_actions, None, Some(Truncation::TurnLimit));
        }
        let transition = if matches!(position.phase, Phase::Chance { .. }) {
            let Some(outcome) = sample_outcome(&position, &mut chance_rng, &mut scratch.outcomes)
            else {
                return result(
                    &position,
                    player_actions,
                    None,
                    Some(Truncation::ActionLimit),
                );
            };
            apply_outcome_checked_with_context(&mut position, context, outcome)
                .expect("enumerated outcome must apply")
        } else {
            if player_actions >= limits.action_limit {
                return result(
                    &position,
                    player_actions,
                    None,
                    Some(Truncation::ActionLimit),
                );
            }
            generate_actions_with_context(&position, context, &mut scratch.actions);
            let Some(action) = choose_action(&scratch.actions, policy, &mut policy_rng) else {
                return result(
                    &position,
                    player_actions,
                    None,
                    Some(Truncation::ActionLimit),
                );
            };
            player_actions += 1;
            let actor = position.actor;
            apply_checked_with_context(&mut position, context, actor, action)
                .expect("generated action must apply")
        };
        match transition.status {
            Status::Won(winner) => {
                return result(&position, player_actions, Some(winner), None);
            }
            Status::Truncated(reason) => {
                return result(&position, player_actions, None, Some(reason));
            }
            Status::Decision | Status::Chance => {}
        }
    }
}

fn debug_assert_conservation(_position: &Position) {
    #[cfg(debug_assertions)]
    {
        for resource in 0..5 {
            let held: u16 = _position.players[..usize::from(_position.player_count)]
                .iter()
                .map(|player| u16::from(player.hand[resource]))
                .sum();
            debug_assert_eq!(held + u16::from(_position.bank[resource]), 19);
        }
        for raw in 0.._position.player_count {
            let player = &_position.players[usize::from(raw)];
            let roads = _position
                .roads
                .iter()
                .filter(|&&owner| owner == raw + 1)
                .count();
            let settlements = _position
                .buildings
                .iter()
                .filter(|&&building| building == raw + 1)
                .count();
            let cities = _position
                .buildings
                .iter()
                .filter(|&&building| building == raw + 1 + catanatron_core::CITY_OFFSET)
                .count();
            debug_assert_eq!(usize::from(player.pieces[0]) + roads, 15);
            debug_assert_eq!(usize::from(player.pieces[1]) + settlements, 5);
            debug_assert_eq!(usize::from(player.pieces[2]) + cities, 4);
        }
    }
}

fn result(
    position: &Position,
    player_actions: u32,
    winner: Option<PlayerId>,
    truncation: Option<Truncation>,
) -> RolloutResult {
    RolloutResult {
        winner,
        truncation,
        turns: position.turns,
        player_actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_base, NumberPlacement};

    #[test]
    fn rollout_is_reproducible_and_does_not_mutate_root() {
        let (context, root) = initialize_base(2, NumberPlacement::OfficialSpiral, 9, 0).unwrap();
        let original = root;
        let limits = RolloutLimits {
            turn_limit: 50,
            action_limit: 5_000,
        };
        let first = rollout(
            &context,
            &root,
            Policy::Weighted,
            9,
            limits,
            &mut RolloutScratch::default(),
        );
        let second = rollout(
            &context,
            &root,
            Policy::Weighted,
            9,
            limits,
            &mut RolloutScratch::default(),
        );
        assert_eq!(first, second);
        assert_eq!(root, original);
        assert!(first.player_actions > 0);
    }

    #[test]
    fn deadline_hook_interrupts_inside_a_rollout() {
        let (context, root) = initialize_base(2, NumberPlacement::OfficialSpiral, 9, 0).unwrap();
        let mut checks = 0;
        let result = rollout_until(
            &context,
            &root,
            Policy::Weighted,
            9,
            RolloutLimits::default(),
            &mut RolloutScratch::default(),
            || {
                checks += 1;
                checks > 2
            },
        );
        assert_eq!(result.truncation, Some(Truncation::ActionLimit));
        assert!(result.player_actions <= 2);
    }
}
