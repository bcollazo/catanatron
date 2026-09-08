use catanatron_core::{
    draw_bounded, enumerate_outcomes, Action, Outcome, RandomSource, WeightedOutcome,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    Random,
    Weighted,
}

pub fn choose_action(
    actions: &[Action],
    policy: Policy,
    random: &mut impl RandomSource,
) -> Option<Action> {
    let total: u64 = actions
        .iter()
        .map(|action| action_weight(*action, policy))
        .sum();
    let mut draw = draw_bounded(random, total)?;
    for &action in actions {
        let weight = action_weight(action, policy);
        if draw < weight {
            return Some(action);
        }
        draw -= weight;
    }
    None
}

fn action_weight(action: Action, policy: Policy) -> u64 {
    if policy == Policy::Random {
        return 1;
    }
    match action {
        Action::BuildCity(_) => 10_000,
        Action::BuildSettlement(_) => 1_000,
        Action::BuyDevelopmentCard => 100,
        _ => 1,
    }
}

pub fn sample_outcome(
    position: &catanatron_core::Position,
    random: &mut impl RandomSource,
    scratch: &mut Vec<WeightedOutcome>,
) -> Option<Outcome> {
    let total = u64::from(enumerate_outcomes(position, scratch));
    let mut draw = draw_bounded(random, total)?;
    for entry in scratch {
        if draw < u64::from(entry.weight) {
            return Some(entry.outcome);
        }
        draw -= u64::from(entry.weight);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(u64);
    impl RandomSource for Fixed {
        fn next_u64(&mut self) -> u64 {
            self.0
        }
    }

    #[test]
    fn weighted_policy_uses_cumulative_weights_without_expansion() {
        let node = catanatron_core::NodeId::new(0).unwrap();
        let actions = [
            Action::EndTurn,
            Action::BuildSettlement(node),
            Action::BuildCity(node),
        ];
        assert_eq!(
            choose_action(&actions, Policy::Weighted, &mut Fixed(11_001)),
            Some(Action::EndTurn)
        );
        assert_eq!(
            choose_action(&actions, Policy::Weighted, &mut Fixed(11_002)),
            Some(Action::BuildSettlement(node))
        );
        assert_eq!(
            choose_action(&actions, Policy::Weighted, &mut Fixed(12_002)),
            Some(Action::BuildCity(node))
        );
    }
}
