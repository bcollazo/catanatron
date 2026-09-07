//! Checked, copy-first transition entry point.
use crate::{validate_boundary, Action, ChanceKind, IllegalAction, Phase, Position, Status};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    pub status: Status,
}

/// Applies a supported intent atomically; failed validation leaves `position` unchanged.
pub fn apply_checked(
    position: &mut Position,
    actor: crate::PlayerId,
    action: Action,
) -> Result<Transition, IllegalAction> {
    validate_boundary(position, actor, action)?;
    let mut next = *position;
    let status = match action {
        Action::Roll => {
            next.phase = Phase::Chance {
                actor,
                kind: ChanceKind::Dice,
            };
            Status::Chance
        }
        Action::EndTurn => {
            let next_actor = crate::PlayerId::new((actor.get() + 1) % next.player_count)
                .expect("active player count bounds next actor");
            next.players[usize::from(actor.get())].played_dev = false;
            next.actor = next_actor;
            next.turn_owner = next_actor;
            next.turns = next.turns.saturating_add(1);
            next.phase = Phase::PreRoll { actor: next_actor };
            Status::Decision
        }
        _ => return Err(IllegalAction::WrongPhase),
    };
    *position = next;
    Ok(Transition { status })
}
