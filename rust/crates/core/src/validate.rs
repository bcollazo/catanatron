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
    InsufficientResource(Resource),
    ExhaustedRoads,
    ExhaustedSettlements,
    ExhaustedCities,
    InvalidRoadPlacement,
    InvalidSettlementPlacement,
    InvalidCityPlacement,
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
            | (Phase::PostRoll { .. }, Action::BuildRoad(_))
            | (Phase::PostRoll { .. }, Action::BuildSettlement(_))
            | (Phase::PostRoll { .. }, Action::BuildCity(_))
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
    Ok(())
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

fn validate_road_placement(
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
        Phase::PostRoll { .. } => {
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
