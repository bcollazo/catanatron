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
        Action::BuildSettlement(node) => match next.phase {
            Phase::SetupSettlement { reverse, .. } => {
                next.buildings[usize::from(node.get())] = actor.get() + 1;
                next.players[usize::from(actor.get())].pieces[1] -= 1;
                next.phase = Phase::SetupRoad {
                    actor,
                    settlement: node,
                    reverse,
                };
                Status::Decision
            }
            Phase::PostRoll { .. } => {
                next.buildings[usize::from(node.get())] = actor.get() + 1;
                let player = &mut next.players[usize::from(actor.get())];
                player.pieces[1] -= 1;
                for resource in [
                    crate::Resource::Wood,
                    crate::Resource::Brick,
                    crate::Resource::Sheep,
                    crate::Resource::Wheat,
                ] {
                    player.hand[resource.index()] -= 1;
                    next.bank[resource.index()] += 1;
                }
                Status::Decision
            }
            _ => return Err(IllegalAction::WrongPhase),
        },
        Action::BuildCity(node) => {
            next.buildings[usize::from(node.get())] = actor.get() + 1 + crate::CITY_OFFSET;
            let player = &mut next.players[usize::from(actor.get())];
            player.pieces[1] += 1;
            player.pieces[2] -= 1;
            for _ in 0..2 {
                player.hand[crate::Resource::Wheat.index()] -= 1;
                next.bank[crate::Resource::Wheat.index()] += 1;
            }
            for _ in 0..3 {
                player.hand[crate::Resource::Ore.index()] -= 1;
                next.bank[crate::Resource::Ore.index()] += 1;
            }
            Status::Decision
        }
        Action::BuyDevelopmentCard => {
            let player = &mut next.players[usize::from(actor.get())];
            for resource in [
                crate::Resource::Sheep,
                crate::Resource::Wheat,
                crate::Resource::Ore,
            ] {
                player.hand[resource.index()] -= 1;
                next.bank[resource.index()] += 1;
            }
            next.phase = Phase::Chance {
                actor,
                kind: ChanceKind::DevelopmentCard,
            };
            Status::Chance
        }
        Action::BuildRoad(edge) => match next.phase {
            Phase::PostRoll { .. } => {
                next.roads[usize::from(edge.get())] = actor.get() + 1;
                let player = &mut next.players[usize::from(actor.get())];
                player.pieces[0] -= 1;
                for resource in [crate::Resource::Wood, crate::Resource::Brick] {
                    player.hand[resource.index()] -= 1;
                    next.bank[resource.index()] += 1;
                }
                Status::Decision
            }
            Phase::SetupRoad { reverse, .. } => {
                next.roads[usize::from(edge.get())] = actor.get() + 1;
                next.players[usize::from(actor.get())].pieces[0] -= 1;
                let settlements = next.buildings.iter().filter(|&&owner| owner != 0).count();
                if settlements == usize::from(next.player_count) * 2 {
                    next.actor = crate::PlayerId::new(0).expect("seat zero");
                    next.turn_owner = next.actor;
                    next.phase = Phase::PreRoll { actor: next.actor };
                } else {
                    let next_raw = if settlements == usize::from(next.player_count) {
                        actor.get()
                    } else if reverse {
                        (actor.get() + next.player_count - 1) % next.player_count
                    } else {
                        (actor.get() + 1) % next.player_count
                    };
                    let next_actor = crate::PlayerId::new(next_raw).expect("active seat");
                    next.actor = next_actor;
                    next.turn_owner = next_actor;
                    next.phase = Phase::SetupSettlement {
                        actor: next_actor,
                        reverse: settlements >= usize::from(next.player_count),
                    };
                }
                Status::Decision
            }
            _ => return Err(IllegalAction::WrongPhase),
        },
        _ => return Err(IllegalAction::WrongPhase),
    };
    *position = next;
    Ok(Transition { status })
}

/// Resolves a checked chance result atomically.
pub fn apply_outcome_checked(
    position: &mut Position,
    outcome: crate::Outcome,
) -> Result<Transition, IllegalAction> {
    crate::validate_outcome(position, outcome)?;
    let mut next = *position;
    let status = match (next.phase, outcome) {
        (
            Phase::Chance {
                actor,
                kind: ChanceKind::DevelopmentCard,
            },
            crate::Outcome::DevelopmentCard(card),
        ) => {
            next.dev_bank[card.index()] -= 1;
            next.players[usize::from(actor.get())].dev[card.index()] += 1;
            next.phase = Phase::PostRoll { actor };
            Status::Decision
        }
        _ => return Err(IllegalAction::InvalidOutcome),
    };
    *position = next;
    Ok(Transition { status })
}
