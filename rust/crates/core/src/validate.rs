//! Non-mutating checked boundary validation shared by later rule transitions.
use crate::{Action, Phase, PlayerId, Position};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IllegalAction {
    InvalidPlayerCount(u8),
    WrongActor {
        expected: PlayerId,
        actual: PlayerId,
    },
    WrongPhase,
    Terminal,
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
        (Phase::PreRoll { .. }, Action::Roll) | (Phase::PostRoll { .. }, Action::EndTurn)
    );
    if allowed {
        Ok(())
    } else {
        Err(IllegalAction::WrongPhase)
    }
}
