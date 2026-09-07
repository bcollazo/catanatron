//! Non-mutating checked boundary validation shared by later rule transitions.
use crate::{
    edge_endpoints, incident, Action, ChanceKind, Outcome, Phase, PlayerId, Position, Resource,
    BASE_EDGE_COUNT,
};
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
    UnsupportedOutcome,
    InsufficientResource(Resource),
    InsufficientBankResource(Resource),
    ExhaustedRoads,
    ExhaustedSettlements,
    ExhaustedCities,
    ExhaustedDevelopmentCards,
    InvalidRoadPlacement,
    InvalidSettlementPlacement,
    InvalidCityPlacement,
    InvalidRobberMove,
    IneligibleDevelopmentCard,
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
                kind: ChanceKind::Theft { victim, .. },
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
            | (Phase::PreRoll { .. }, Action::PlayKnight)
            | (Phase::PreRoll { .. }, Action::YearOfPlenty { .. })
            | (Phase::PreRoll { .. }, Action::Monopoly(_))
            | (Phase::PreRoll { .. }, Action::RoadBuilding)
            | (Phase::PostRoll { .. }, Action::EndTurn)
            | (Phase::PostRoll { .. }, Action::BuildRoad(_))
            | (Phase::PostRoll { .. }, Action::BuildSettlement(_))
            | (Phase::PostRoll { .. }, Action::BuildCity(_))
            | (Phase::PostRoll { .. }, Action::BuyDevelopmentCard)
            | (Phase::PostRoll { .. }, Action::PlayKnight)
            | (Phase::PostRoll { .. }, Action::YearOfPlenty { .. })
            | (Phase::PostRoll { .. }, Action::Monopoly(_))
            | (Phase::PostRoll { .. }, Action::RoadBuilding)
            | (Phase::Discard { .. }, Action::Discard(_))
            | (Phase::Robber { .. }, Action::MoveRobber { .. })
            | (Phase::FreeRoad { .. }, Action::BuildRoad(_))
    );
    if !allowed {
        return Err(IllegalAction::WrongPhase);
    }
    if let Action::BuildRoad(edge) = action {
        validate_road_placement(position, actor, edge)?;
        if matches!(position.phase, Phase::PostRoll { .. }) {
            for resource in [Resource::Wood, Resource::Brick] {
                if position.players[usize::from(actor.get())].hand[resource.index()] == 0 {
                    return Err(IllegalAction::InsufficientResource(resource));
                }
            }
        }
    }
    if let Action::BuildSettlement(node) = action {
        validate_settlement_placement(position, actor, node)?;
        if matches!(position.phase, Phase::PostRoll { .. }) {
            for resource in [
                Resource::Wood,
                Resource::Brick,
                Resource::Sheep,
                Resource::Wheat,
            ] {
                if position.players[usize::from(actor.get())].hand[resource.index()] == 0 {
                    return Err(IllegalAction::InsufficientResource(resource));
                }
            }
        }
    }
    if let Action::BuildCity(node) = action {
        if position.players[usize::from(actor.get())].pieces[2] == 0 {
            return Err(IllegalAction::ExhaustedCities);
        }
        if position.buildings[usize::from(node.get())] != actor.get() + 1 {
            return Err(IllegalAction::InvalidCityPlacement);
        }
        for (resource, required) in [(Resource::Wheat, 2), (Resource::Ore, 3)] {
            if position.players[usize::from(actor.get())].hand[resource.index()] < required {
                return Err(IllegalAction::InsufficientResource(resource));
            }
        }
    }
    if matches!(action, Action::BuyDevelopmentCard) {
        if position.dev_bank.iter().all(|&count| count == 0) {
            return Err(IllegalAction::ExhaustedDevelopmentCards);
        }
        for resource in [Resource::Sheep, Resource::Wheat, Resource::Ore] {
            if position.players[usize::from(actor.get())].hand[resource.index()] == 0 {
                return Err(IllegalAction::InsufficientResource(resource));
            }
        }
    }
    if let Action::Discard(resource) = action {
        if position.players[usize::from(actor.get())].hand[resource.index()] == 0 {
            return Err(IllegalAction::InsufficientResource(resource));
        }
    }
    if let Action::MoveRobber { tile, victim } = action {
        if tile.get() == position.robber {
            return Err(IllegalAction::InvalidRobberMove);
        }
        if let Some(victim) = victim {
            if victim == actor
                || victim.get() >= position.player_count
                || position.players[usize::from(victim.get())]
                    .hand
                    .iter()
                    .all(|&count| count == 0)
                || !crate::land_tile_nodes(tile).into_iter().any(|node| {
                    crate::building_belongs_to(position.buildings[usize::from(node.get())], victim)
                })
            {
                return Err(IllegalAction::InvalidRobberMove);
            }
        } else if crate::land_tile_nodes(tile).into_iter().any(|node| {
            crate::building_owner(position.buildings[usize::from(node.get())]).is_some_and(
                |owner| {
                    owner != actor
                        && position.players[usize::from(owner.get())]
                            .hand
                            .iter()
                            .any(|&count| count > 0)
                },
            )
        }) {
            return Err(IllegalAction::InvalidRobberMove);
        }
    }
    if matches!(action, Action::PlayKnight) {
        validate_development_card(position, actor, crate::DevelopmentCard::Knight)?;
    }
    if let Action::YearOfPlenty { first, second } = action {
        validate_development_card(position, actor, crate::DevelopmentCard::YearOfPlenty)?;
        if position.bank[first.index()] == 0 {
            return Err(IllegalAction::InsufficientBankResource(first));
        }
        if let Some(second) = second {
            let required = if first == second { 2 } else { 1 };
            if position.bank[second.index()] < required {
                return Err(IllegalAction::InsufficientBankResource(second));
            }
        }
    }
    if matches!(action, Action::Monopoly(_)) {
        validate_development_card(position, actor, crate::DevelopmentCard::Monopoly)?;
    }
    if matches!(action, Action::RoadBuilding) {
        validate_development_card(position, actor, crate::DevelopmentCard::RoadBuilding)?;
        let mut probe = *position;
        probe.phase = Phase::FreeRoad {
            actor,
            remaining: 2,
            resume_post_roll: matches!(position.phase, Phase::PostRoll { .. }),
        };
        if !(0..crate::BASE_EDGE_COUNT as u8).any(|raw| {
            validate_road_placement(&probe, actor, crate::EdgeId::new(raw).expect("base edge"))
                .is_ok()
        }) {
            return Err(IllegalAction::InvalidRoadPlacement);
        }
    }
    Ok(())
}

fn validate_development_card(
    position: &Position,
    actor: PlayerId,
    card: crate::DevelopmentCard,
) -> Result<(), IllegalAction> {
    let player = &position.players[usize::from(actor.get())];
    if player.played_dev
        || player.dev[card.index()] == 0
        || player.eligible_dev_mask & (1 << card.index()) == 0
    {
        Err(IllegalAction::IneligibleDevelopmentCard)
    } else {
        Ok(())
    }
}

fn validate_settlement_placement(
    position: &Position,
    actor: PlayerId,
    node: crate::NodeId,
) -> Result<(), IllegalAction> {
    if position.players[usize::from(actor.get())].pieces[1] == 0 {
        return Err(IllegalAction::ExhaustedSettlements);
    }
    if position.buildings[usize::from(node.get())] != 0
        || crate::node_neighbors(node).any(|near| position.buildings[usize::from(near.get())] != 0)
    {
        return Err(IllegalAction::InvalidSettlementPlacement);
    }
    if matches!(position.phase, Phase::PostRoll { .. })
        && !(0..BASE_EDGE_COUNT as u8).any(|raw| {
            position.roads[usize::from(raw)] == actor.get() + 1
                && incident(crate::EdgeId::new(raw).expect("base edge"), node)
        })
    {
        return Err(IllegalAction::InvalidSettlementPlacement);
    }
    Ok(())
}

pub(crate) fn validate_road_placement(
    position: &Position,
    actor: PlayerId,
    edge: crate::EdgeId,
) -> Result<(), IllegalAction> {
    if position.players[usize::from(actor.get())].pieces[0] == 0 {
        return Err(IllegalAction::ExhaustedRoads);
    }
    if position.roads[usize::from(edge.get())] != 0 {
        return Err(IllegalAction::InvalidRoadPlacement);
    }
    let connected = match position.phase {
        Phase::SetupRoad { settlement, .. } => incident(edge, settlement),
        Phase::PostRoll { .. } | Phase::FreeRoad { .. } => {
            let (first, second) = edge_endpoints(edge);
            [first, second].into_iter().any(|node| {
                let building = position.buildings[usize::from(node.get())];
                crate::building_belongs_to(building, actor)
                    || (building == 0
                        && (0..BASE_EDGE_COUNT as u8).any(|raw| {
                            position.roads[usize::from(raw)] == actor.get() + 1
                                && incident(crate::EdgeId::new(raw).expect("base edge"), node)
                        }))
            })
        }
        _ => false,
    };
    if connected {
        Ok(())
    } else {
        Err(IllegalAction::InvalidRoadPlacement)
    }
}
