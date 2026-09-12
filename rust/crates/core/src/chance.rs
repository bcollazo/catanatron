//! Exact integer-weight chance enumeration; sampling belongs to search.
use crate::{ChanceKind, DevelopmentCard, Outcome, Phase, Position, Resource};

pub trait RandomSource {
    fn next_u64(&mut self) -> u64;
}

pub fn draw_bounded(random: &mut impl RandomSource, bound: u64) -> Option<u64> {
    if bound == 0 {
        return None;
    }
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = random.next_u64();
        if value >= threshold {
            return Some(value % bound);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightedOutcome {
    pub outcome: Outcome,
    pub weight: u16,
}

pub fn enumerate_outcomes(position: &Position, out: &mut Vec<WeightedOutcome>) -> u16 {
    out.clear();
    let Phase::Chance { kind, .. } = position.phase else {
        return 0;
    };
    match kind {
        ChanceKind::Dice => {
            for first in 1..=6 {
                for second in 1..=6 {
                    out.push(WeightedOutcome {
                        outcome: Outcome::Dice { first, second },
                        weight: 1,
                    });
                }
            }
        }
        ChanceKind::Theft { victim, .. } => {
            for resource in Resource::ALL {
                let weight =
                    u16::from(position.players[usize::from(victim.get())].hand[resource.index()]);
                if weight > 0 {
                    out.push(WeightedOutcome {
                        outcome: Outcome::StolenResource(resource),
                        weight,
                    });
                }
            }
        }
        ChanceKind::DevelopmentCard => {
            for card in DevelopmentCard::ALL {
                let weight = u16::from(position.dev_bank[card.index()]);
                if weight > 0 {
                    out.push(WeightedOutcome {
                        outcome: Outcome::DevelopmentCard(card),
                        weight,
                    });
                }
            }
        }
    }
    out.iter().map(|entry| entry.weight).sum()
}
