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
        Action::Discard(resource) => {
            let remaining = match next.phase {
                Phase::Discard { remaining, .. } => remaining,
                _ => return Err(IllegalAction::WrongPhase),
            };
            next.players[usize::from(actor.get())].hand[resource.index()] -= 1;
            next.bank[resource.index()] += 1;
            if remaining > 1 {
                next.phase = Phase::Discard {
                    actor,
                    remaining: remaining - 1,
                };
            } else if let Some(next_actor) = next_discarder(&next, actor.get() + 1) {
                next.actor = next_actor;
                next.phase = Phase::Discard {
                    actor: next_actor,
                    remaining: discard_count(&next, next_actor),
                };
            } else {
                next.actor = next.turn_owner;
                next.phase = Phase::Robber {
                    actor: next.turn_owner,
                    resume_post_roll: true,
                };
            }
            Status::Decision
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
                    if settlements != usize::from(next.player_count) {
                        next.turns = next.turns.saturating_add(1);
                    }
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

fn discard_count(position: &Position, player: crate::PlayerId) -> u8 {
    let total: u8 = position.players[usize::from(player.get())]
        .hand
        .iter()
        .sum();
    if total > 7 {
        total / 2
    } else {
        0
    }
}

fn next_discarder(position: &Position, first: u8) -> Option<crate::PlayerId> {
    (first..position.player_count).find_map(|raw| {
        let player = crate::PlayerId::new(raw).expect("active player");
        (discard_count(position, player) > 0).then_some(player)
    })
}

/// Applies an intent using immutable board assignments when a rule needs them.
///
/// The context-free entry point remains useful for geometry-only callers. New
/// game transitions should use this entry point so setup and production rules
/// observe the concrete randomized layout.
pub fn apply_checked_with_context(
    position: &mut Position,
    context: &crate::GameContext,
    actor: crate::PlayerId,
    action: Action,
) -> Result<Transition, IllegalAction> {
    let mut next = *position;
    let transition = apply_checked(&mut next, actor, action)?;
    if let Action::BuildSettlement(node) = action {
        if matches!(position.phase, Phase::SetupSettlement { .. })
            && next
                .buildings
                .iter()
                .filter(|&&building| crate::building_belongs_to(building, actor))
                .count()
                == 2
        {
            for raw in 0..crate::BASE_LAND_TILE_COUNT as u8 {
                let tile = crate::TileId::new(raw).expect("generated tile");
                if crate::land_tile_nodes(tile).contains(&node) {
                    if let Some(resource) = context.layout.tile(tile).resource {
                        next.bank[resource.index()] -= 1;
                        next.players[usize::from(actor.get())].hand[resource.index()] += 1;
                    }
                }
            }
        }
    }
    *position = next;
    Ok(transition)
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

/// Resolves chance using immutable layout assignments where production needs them.
pub fn apply_outcome_checked_with_context(
    position: &mut Position,
    context: &crate::GameContext,
    outcome: crate::Outcome,
) -> Result<Transition, IllegalAction> {
    if !matches!(outcome, crate::Outcome::Dice { .. }) {
        return apply_outcome_checked(position, outcome);
    }
    crate::validate_outcome(position, outcome)?;
    let crate::Outcome::Dice { first, second } = outcome else {
        unreachable!("dice outcome was matched above");
    };
    let Phase::Chance {
        actor,
        kind: ChanceKind::Dice,
    } = position.phase
    else {
        return Err(IllegalAction::InvalidOutcome);
    };
    if first + second == 7 {
        let mut next = *position;
        if let Some(discarder) = next_discarder(&next, 0) {
            next.actor = discarder;
            next.phase = Phase::Discard {
                actor: discarder,
                remaining: discard_count(&next, discarder),
            };
        } else {
            next.actor = next.turn_owner;
            next.phase = Phase::Robber {
                actor: next.turn_owner,
                resume_post_roll: true,
            };
        }
        *position = next;
        return Ok(Transition {
            status: Status::Decision,
        });
    }
    let mut next = *position;
    let mut demand = [[0_u8; 5]; crate::MAX_PLAYERS];
    for raw in 0..crate::BASE_LAND_TILE_COUNT as u8 {
        if raw == next.robber {
            continue;
        }
        let tile = crate::TileId::new(raw).expect("generated tile");
        let assignment = context.layout.tile(tile);
        if assignment.number != Some(first + second) {
            continue;
        }
        let Some(resource) = assignment.resource else {
            continue;
        };
        for node in crate::land_tile_nodes(tile) {
            let building = next.buildings[usize::from(node.get())];
            if let Some(owner) = crate::building_owner(building) {
                demand[usize::from(owner.get())][resource.index()] +=
                    crate::building_production(building);
            }
        }
    }
    for resource in crate::Resource::ALL {
        let total: u8 = demand.iter().map(|player| player[resource.index()]).sum();
        if total <= next.bank[resource.index()] {
            next.bank[resource.index()] -= total;
            for (index, player) in next.players.iter_mut().enumerate() {
                player.hand[resource.index()] += demand[index][resource.index()];
            }
        }
    }
    next.phase = Phase::PostRoll { actor };
    *position = next;
    Ok(Transition {
        status: Status::Decision,
    })
}
