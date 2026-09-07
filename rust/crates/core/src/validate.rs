//! Non-mutating checked boundary validation shared by later rule transitions.
use crate::{Action, ChanceKind, Outcome, Phase, PlayerId, Position};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IllegalAction {
    InvalidPlayerCount(u8),
    WrongActor {
        expected: PlayerId,
        actual: PlayerId,
    },
    WrongPhase,
    Terminal,
    InvalidOutcome,
}

/// Checks a supplied chance result before an application path can mutate state.
pub fn validate_outcome(position: &Position, outcome: Outcome) -> Result<(), IllegalAction> {
    match (position.phase, outcome) {
        (
            Phase::Chance {
                kind: ChanceKind::Dice,
                ..
            },
            Outcome::Dice { first, second },
        ) if (1..=6).contains(&first) && (1..=6).contains(&second) => Ok(()),
        (
            Phase::Chance {
                kind: ChanceKind::Theft { victim },
                ..
            },
            Outcome::StolenResource(resource),
        ) if position.players[usize::from(victim.get())].hand[resource.index()] > 0 => Ok(()),
        (
            Phase::Chance {
                kind: ChanceKind::DevelopmentCard,
                ..
            },
            Outcome::DevelopmentCard(card),
        ) if position.dev_bank[card.index()] > 0 => Ok(()),
        _ => Err(IllegalAction::InvalidOutcome),
    }
}
pub fn validate_boundary(
    position: &Position,
    actor: PlayerId,
    action: Action,
) -> Result<(), IllegalAction> {
    if actor.get() >= position.player_count {
        return Err(IllegalAction::WrongActor {
            expected: position.actor,
            actual: actor,
        });
    }
    if matches!(position.phase, Phase::Terminal) {
        return Err(IllegalAction::Terminal);
    }
    if actor != position.actor {
        return Err(IllegalAction::WrongActor {
            expected: position.actor,
            actual: actor,
        });
    }
    let allowed = matches!(
        (position.phase, action),
        (Phase::SetupSettlement { .. }, Action::BuildSettlement(_))
            | (Phase::SetupRoad { .. }, Action::BuildRoad(_))
            | (Phase::PreRoll { .. }, Action::Roll)
            | (Phase::PostRoll { .. }, Action::EndTurn)
    );
    if allowed {
        Ok(())
    } else {
        Err(IllegalAction::WrongPhase)
    }
}
