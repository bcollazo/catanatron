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
            next.players[usize::from(actor.get())].eligible_dev_mask = next.players
                [usize::from(actor.get())]
            .dev
            .iter()
            .take(4)
            .enumerate()
            .fold(0, |mask, (index, &count)| {
                mask | ((count > 0) as u8) << index
            });
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
        Action::MoveRobber { tile, victim } => {
            let resume_post_roll = match next.phase {
                Phase::Robber {
                    resume_post_roll, ..
                } => resume_post_roll,
                _ => return Err(IllegalAction::WrongPhase),
            };
            next.robber = tile.get();
            if let Some(victim) = victim {
                next.phase = Phase::Chance {
                    actor,
                    kind: ChanceKind::Theft {
                        victim,
                        resume_post_roll,
                    },
                };
                Status::Chance
            } else {
                next.phase = if resume_post_roll {
                    Phase::PostRoll { actor }
                } else {
                    Phase::PreRoll { actor }
                };
                Status::Decision
            }
        }
        Action::PlayKnight => {
            let resume_post_roll = matches!(next.phase, Phase::PostRoll { .. });
            consume_development_card(&mut next, actor, crate::DevelopmentCard::Knight);
            next.players[usize::from(actor.get())].played_knights += 1;
            next.phase = Phase::Robber {
                actor,
                resume_post_roll,
            };
            Status::Decision
        }
        Action::YearOfPlenty { first, second } => {
            next.bank[first.index()] -= 1;
            next.players[usize::from(actor.get())].hand[first.index()] += 1;
            if let Some(second) = second {
                next.bank[second.index()] -= 1;
                next.players[usize::from(actor.get())].hand[second.index()] += 1;
            }
            consume_development_card(&mut next, actor, crate::DevelopmentCard::YearOfPlenty);
            Status::Decision
        }
        Action::Monopoly(resource) => {
            let mut stolen = 0_u8;
            for raw in 0..next.player_count {
                if raw != actor.get() {
                    stolen += next.players[usize::from(raw)].hand[resource.index()];
                    next.players[usize::from(raw)].hand[resource.index()] = 0;
                }
            }
            next.players[usize::from(actor.get())].hand[resource.index()] += stolen;
            consume_development_card(&mut next, actor, crate::DevelopmentCard::Monopoly);
            Status::Decision
        }
        Action::RoadBuilding => {
            let resume_post_roll = matches!(next.phase, Phase::PostRoll { .. });
            consume_development_card(&mut next, actor, crate::DevelopmentCard::RoadBuilding);
            next.phase = Phase::FreeRoad {
                actor,
                remaining: 2,
                resume_post_roll,
            };
            Status::Decision
        }
        Action::OfferTrade { give, receive } => {
            next.trade_give = give;
            next.trade_receive = receive;
            next.trade_proposer = actor;
            next.trade_responded_mask = 1 << actor.get();
            next.trade_accepted_mask = 0;
            let responder = next_trade_responder(&next, 0).expect("at least two players");
            next.actor = responder;
            next.phase = Phase::TradeResponse { actor: responder };
            Status::Decision
        }
        Action::AcceptTrade | Action::RejectTrade => {
            next.trade_responded_mask |= 1 << actor.get();
            if matches!(action, Action::AcceptTrade) {
                next.trade_accepted_mask |= 1 << actor.get();
            }
            if let Some(responder) = next_trade_responder(&next, actor.get() + 1) {
                next.actor = responder;
                next.phase = Phase::TradeResponse { actor: responder };
            } else {
                next.actor = next.trade_proposer;
                next.phase = if next.trade_accepted_mask == 0 {
                    clear_trade(&mut next);
                    Phase::PostRoll { actor: next.actor }
                } else {
                    Phase::ChooseAccepter { actor: next.actor }
                };
            }
            Status::Decision
        }
        Action::ConfirmTrade(accepter) => {
            for index in 0..5 {
                next.players[usize::from(actor.get())].hand[index] -= next.trade_give[index];
                next.players[usize::from(accepter.get())].hand[index] += next.trade_give[index];
                next.players[usize::from(accepter.get())].hand[index] -= next.trade_receive[index];
                next.players[usize::from(actor.get())].hand[index] += next.trade_receive[index];
            }
            clear_trade(&mut next);
            next.phase = Phase::PostRoll { actor };
            Status::Decision
        }
        Action::CancelTrade => {
            clear_trade(&mut next);
            next.phase = Phase::PostRoll { actor };
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
            Phase::FreeRoad {
                remaining,
                resume_post_roll,
                ..
            } => {
                next.roads[usize::from(edge.get())] = actor.get() + 1;
                next.players[usize::from(actor.get())].pieces[0] -= 1;
                let remaining = remaining - 1;
                next.phase = Phase::FreeRoad {
                    actor,
                    remaining,
                    resume_post_roll,
                };
                let can_continue = remaining > 0
                    && (0..crate::BASE_EDGE_COUNT as u8).any(|raw| {
                        crate::validate::validate_road_placement(
                            &next,
                            actor,
                            crate::EdgeId::new(raw).expect("base edge"),
                        )
                        .is_ok()
                    });
                if !can_continue {
                    next.phase = if resume_post_roll {
                        Phase::PostRoll { actor }
                    } else {
                        Phase::PreRoll { actor }
                    };
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
    let status = finalize_transition(&mut next, status);
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

fn consume_development_card(
    position: &mut Position,
    actor: crate::PlayerId,
    card: crate::DevelopmentCard,
) {
    let player = &mut position.players[usize::from(actor.get())];
    player.dev[card.index()] -= 1;
    // Python's OWNED_AT_START flags are a turn-start snapshot. They remain
    // set after the last eligible card is consumed and are refreshed on end
    // turn; `dev` and `played_dev` still prevent a second play.
    player.played_dev = true;
}

fn next_trade_responder(position: &Position, first: u8) -> Option<crate::PlayerId> {
    (first..position.player_count).find_map(|raw| {
        let mask = 1 << raw;
        if position.trade_responded_mask & mask == 0 {
            crate::PlayerId::new(raw).ok()
        } else {
            None
        }
    })
}

fn clear_trade(position: &mut Position) {
    position.trade_give = [0; 5];
    position.trade_receive = [0; 5];
    position.trade_responded_mask = 0;
    position.trade_accepted_mask = 0;
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
    if matches!(action, Action::MoveRobber { .. }) && context.friendly_robber {
        let mut legal = Vec::new();
        crate::generate_actions_with_context(position, context, &mut legal);
        if !legal.contains(&action) {
            return Err(IllegalAction::InvalidRobberMove);
        }
    }
    if let Action::MaritimeTrade {
        give,
        receive,
        rate,
    } = action
    {
        if actor != position.actor || !matches!(position.phase, Phase::PostRoll { .. }) {
            return Err(IllegalAction::WrongPhase);
        }
        let expected_rate = crate::maritime_rate(position, context, actor, give);
        if give == receive
            || rate != expected_rate
            || position.players[usize::from(actor.get())].hand[give.index()] < rate
            || position.bank[receive.index()] == 0
        {
            return Err(IllegalAction::InvalidTrade);
        }
        let mut next = *position;
        next.players[usize::from(actor.get())].hand[give.index()] -= rate;
        next.bank[give.index()] += rate;
        next.bank[receive.index()] -= 1;
        next.players[usize::from(actor.get())].hand[receive.index()] += 1;
        let status = finalize_transition(&mut next, Status::Decision);
        *position = next;
        return Ok(Transition { status });
    }
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
    let status = finalize_transition(&mut next, transition.status);
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
        (
            Phase::Chance {
                actor,
                kind:
                    ChanceKind::Theft {
                        victim,
                        resume_post_roll,
                    },
            },
            crate::Outcome::StolenResource(resource),
        ) => {
            next.players[usize::from(victim.get())].hand[resource.index()] -= 1;
            next.players[usize::from(actor.get())].hand[resource.index()] += 1;
            next.phase = if resume_post_roll {
                Phase::PostRoll { actor }
            } else {
                Phase::PreRoll { actor }
            };
            Status::Decision
        }
        _ => return Err(IllegalAction::InvalidOutcome),
    };
    let status = finalize_transition(&mut next, status);
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
        let status = finalize_transition(&mut next, Status::Decision);
        *position = next;
        return Ok(Transition { status });
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
    let status = finalize_transition(&mut next, Status::Decision);
    *position = next;
    Ok(Transition { status })
}

fn finalize_transition(position: &mut Position, provisional: Status) -> Status {
    if provisional == Status::Chance {
        return provisional;
    }
    crate::awards::refresh_awards(position);
    let mut winner = None;
    for raw in 0..position.player_count {
        let player = crate::PlayerId::new(raw).expect("active player");
        if crate::actual_victory_points(position, player) >= 10 {
            winner = Some(player);
        }
    }
    if let Some(winner) = winner {
        position.phase = Phase::Terminal;
        Status::Won(winner)
    } else if position.turns >= 1_000 {
        position.phase = Phase::Terminal;
        Status::Truncated(crate::Truncation::TurnLimit)
    } else {
        provisional
    }
}
