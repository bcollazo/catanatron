use catanatron_core::{
    apply_checked_with_context, apply_outcome_checked_with_context, Action, GameContext, Phase,
    PlayerId, Position, Status,
};

use crate::{rollout_until, sample_outcome, Policy, RolloutLimits, RolloutScratch, SearchRng};

#[derive(Clone, Debug, PartialEq)]
pub struct FlatResult {
    pub action: Action,
    pub mean: f64,
    pub samples: u32,
}

pub fn flat_monte_carlo(
    context: &GameContext,
    root: &Position,
    actions: &[Action],
    simulations: u32,
    seed: u64,
    limits: RolloutLimits,
) -> Option<FlatResult> {
    flat_monte_carlo_until(context, root, actions, simulations, seed, limits, || false)
}

pub fn flat_monte_carlo_until(
    context: &GameContext,
    root: &Position,
    actions: &[Action],
    simulations: u32,
    seed: u64,
    limits: RolloutLimits,
    mut should_stop: impl FnMut() -> bool,
) -> Option<FlatResult> {
    if actions.is_empty() {
        return None;
    }
    let mut sums = vec![0.0; actions.len()];
    let mut counts = vec![0_u32; actions.len()];
    let mut scratch = RolloutScratch::default();
    for sample in 0..simulations.max(actions.len() as u32) {
        if sample >= actions.len() as u32 && should_stop() {
            break;
        }
        let index = sample as usize % actions.len();
        sums[index] += evaluate(
            context,
            root,
            actions[index],
            seed.wrapping_add(u64::from(sample)),
            limits,
            &mut scratch,
            &mut should_stop,
        );
        counts[index] += 1;
    }
    let best = (0..actions.len()).max_by(|&left, &right| {
        let left_mean = sums[left] / f64::from(counts[left]);
        let right_mean = sums[right] / f64::from(counts[right]);
        left_mean
            .total_cmp(&right_mean)
            .then_with(|| format!("{:?}", actions[right]).cmp(&format!("{:?}", actions[left])))
    })?;
    Some(FlatResult {
        action: actions[best],
        mean: sums[best] / f64::from(counts[best]),
        samples: counts[best],
    })
}

fn evaluate(
    context: &GameContext,
    root: &Position,
    action: Action,
    seed: u64,
    limits: RolloutLimits,
    scratch: &mut RolloutScratch,
    should_stop: &mut impl FnMut() -> bool,
) -> f64 {
    let root_player = root.actor;
    let mut position = *root;
    let transition = apply_checked_with_context(&mut position, context, root_player, action)
        .expect("root action came from generated menu");
    if let Some(reward) = status_reward(transition.status, root_player) {
        return reward;
    }
    if matches!(position.phase, Phase::Chance { .. }) {
        let mut outcomes = Vec::with_capacity(36);
        let mut rng =
            SearchRng::from_seed(crate::derive_seed(seed, 0, 0, crate::StreamKind::Chance));
        let Some(outcome) = sample_outcome(&position, &mut rng, &mut outcomes) else {
            return 0.5;
        };
        let transition = apply_outcome_checked_with_context(&mut position, context, outcome)
            .expect("sampled immediate outcome must apply");
        if let Some(reward) = status_reward(transition.status, root_player) {
            return reward;
        }
    }
    let result = rollout_until(
        context,
        &position,
        Policy::Weighted,
        seed,
        limits,
        scratch,
        should_stop,
    );
    result
        .winner
        .map_or(0.5, |winner| if winner == root_player { 1.0 } else { 0.0 })
}

fn status_reward(status: Status, root: PlayerId) -> Option<f64> {
    match status {
        Status::Won(winner) => Some(if winner == root { 1.0 } else { 0.0 }),
        Status::Truncated(_) => Some(0.5),
        Status::Decision | Status::Chance => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_base, NumberPlacement};
    use catanatron_core::generate_actions_with_context;

    #[test]
    fn fixed_search_is_reproducible_and_preserves_root() {
        let (context, root) = initialize_base(2, NumberPlacement::OfficialSpiral, 7, 0).unwrap();
        let original = root;
        let mut actions = Vec::new();
        generate_actions_with_context(&root, &context, &mut actions);
        let a = flat_monte_carlo(&context, &root, &actions, 8, 3, RolloutLimits::default());
        let b = flat_monte_carlo(&context, &root, &actions, 8, 3, RolloutLimits::default());
        assert_eq!(a, b);
        assert_eq!(root, original);
        assert!(actions.contains(&a.unwrap().action));
    }
}
