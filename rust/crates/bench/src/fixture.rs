use catanatron_core::{
    edge_endpoints, DevelopmentCard, EdgeId, GameContext, LandTile, Layout, NodeId, Phase,
    PlayerId, Port, Position, Resource, BASE_EDGE_COUNT, BASE_LAND_TILE_COUNT, BASE_NODE_COUNT,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct FixtureRecord {
    pub case_id: String,
    pub before: FixtureState,
    pub after: FixtureState,
    pub action: WireAction,
    pub outcome: Option<serde_json::Value>,
    pub legal_before: Vec<WireAction>,
    pub legal_after: Vec<WireAction>,
    pub status_after: String,
}

#[derive(Debug, Deserialize)]
pub struct WireAction {
    pub color: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct FixtureState {
    pub colors: Vec<String>,
    pub players: Vec<FixturePlayer>,
    pub bank: [u8; 5],
    pub development_deck: Vec<String>,
    pub buildings: Vec<Option<(String, String)>>,
    pub roads: Vec<([u8; 2], String)>,
    pub ports: Vec<FixturePort>,
    pub actor: String,
    pub turn_owner: String,
    pub prompt: String,
    pub phase: FixturePhase,
    pub initial_build: bool,
    pub road_building: u8,
    pub discard_counts: Vec<u8>,
    pub layout: Vec<FixtureTile>,
    pub robber: [i8; 3],
    pub trade: Vec<serde_json::Value>,
    pub acceptees: Vec<bool>,
    pub responded: Vec<bool>,
    pub turns: u16,
    pub friendly_robber: bool,
}

#[derive(Debug, Deserialize)]
pub struct FixturePlayer {
    pub hand: [u8; 5],
    pub dev: [u8; 5],
    pub eligible_dev: [bool; 4],
    pub pieces: [u8; 3],
    pub played_dev: bool,
    pub played_knights: u8,
    pub longest_road_length: u8,
    pub has_longest_road: bool,
    pub has_largest_army: bool,
    pub has_rolled: bool,
}

#[derive(Debug, Deserialize)]
pub struct FixturePort {
    pub resource: Option<String>,
    pub nodes: [u8; 2],
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FixturePhase {
    SetupSettlement {
        actor: String,
        reverse: bool,
    },
    SetupRoad {
        actor: String,
        settlement: u8,
        reverse: bool,
    },
    PreRoll {
        actor: String,
    },
    PostRoll {
        actor: String,
    },
    Discard {
        actor: String,
        remaining: u8,
    },
    Robber {
        actor: String,
        resume_post_roll: bool,
    },
    FreeRoad {
        actor: String,
        remaining: u8,
        resume_post_roll: bool,
    },
    TradeResponse {
        actor: String,
    },
    ChooseAccepter {
        actor: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct FixtureTile {
    pub coordinate: [i8; 3],
    pub resource: Option<String>,
    pub number: Option<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImportError {
    InvalidPlayerCount,
    UnknownColor(String),
    UnknownResource(String),
    UnknownDevelopmentCard(String),
    InvalidBuilding,
    InvalidRoad([u8; 2]),
    InvalidPort,
    InvalidTrade,
    InvalidLayout,
    InvalidRobber,
    InconsistentActor,
}

pub fn import_state(state: &FixtureState) -> Result<(GameContext, Position), ImportError> {
    if state.colors.len() != state.players.len() || !(2..=4).contains(&state.colors.len()) {
        return Err(ImportError::InvalidPlayerCount);
    }
    let mut position =
        Position::new(state.colors.len() as u8).map_err(|_| ImportError::InvalidPlayerCount)?;
    position.bank = state.bank;
    position.dev_bank = [0; 5];
    for card in &state.development_deck {
        position.dev_bank[development(card)?.index()] += 1;
    }
    for (index, fixture) in state.players.iter().enumerate() {
        let player = &mut position.players[index];
        player.hand = fixture.hand;
        player.dev = fixture.dev;
        player.pieces = fixture.pieces;
        player.played_dev = fixture.played_dev;
        player.played_knights = fixture.played_knights;
        position.longest_road_lengths[index] = fixture.longest_road_length;
        if fixture.has_longest_road {
            position.longest_road_holder = PlayerId::new(index as u8).ok();
        }
        if fixture.has_largest_army {
            position.largest_army_holder = PlayerId::new(index as u8).ok();
        }
        player.eligible_dev_mask = fixture
            .eligible_dev
            .iter()
            .enumerate()
            .fold(0, |mask, (bit, eligible)| {
                mask | (u8::from(*eligible) << bit)
            });
    }
    if state.buildings.len() != BASE_NODE_COUNT {
        return Err(ImportError::InvalidBuilding);
    }
    for (node, building) in state.buildings.iter().enumerate() {
        if let Some((color, kind)) = building {
            let owner = seat(state, color)?;
            position.buildings[node] = match kind.as_str() {
                "SETTLEMENT" => owner.get() + 1,
                "CITY" => owner.get() + 1 + catanatron_core::CITY_OFFSET,
                _ => return Err(ImportError::InvalidBuilding),
            };
        }
    }
    for (road, color) in &state.roads {
        let edge = (0..BASE_EDGE_COUNT as u8)
            .map(|raw| EdgeId::new(raw).expect("base edge"))
            .find(|edge| {
                let (a, b) = edge_endpoints(*edge);
                [a.get(), b.get()] == *road || [b.get(), a.get()] == *road
            })
            .ok_or(ImportError::InvalidRoad(*road))?;
        let owner = seat(state, color)?;
        position.roads[usize::from(edge.get())] = owner.get() + 1;
    }
    let actor = seat(state, &state.actor)?;
    let turn_owner = seat(state, &state.turn_owner)?;
    position.actor = actor;
    position.turn_owner = turn_owner;
    position.turns = state.turns;
    if state.trade.len() != 11 {
        return Err(ImportError::InvalidTrade);
    }
    for index in 0..5 {
        position.trade_give[index] = wire_u8(&state.trade[index])?;
        position.trade_receive[index] = wire_u8(&state.trade[index + 5])?;
    }
    position.trade_proposer = match state.trade[10].as_str() {
        Some(color) => seat(state, color)?,
        None if state.trade[10].as_u64() == Some(0) => PlayerId::new(0).expect("seat zero"),
        None => return Err(ImportError::InvalidTrade),
    };
    position.trade_responded_mask = bool_mask(&state.responded);
    position.trade_accepted_mask = bool_mask(&state.acceptees);
    position.phase = import_phase(state, actor)?;

    if state.layout.len() != BASE_LAND_TILE_COUNT {
        return Err(ImportError::InvalidLayout);
    }
    let mut tiles = [LandTile::DESERT; BASE_LAND_TILE_COUNT];
    for (index, tile) in state.layout.iter().enumerate() {
        tiles[index] = match (&tile.resource, tile.number) {
            (None, None) => LandTile::DESERT,
            (Some(resource), Some(number)) => LandTile::producing(resource_from(resource)?, number),
            _ => return Err(ImportError::InvalidLayout),
        };
    }
    position.robber = state
        .layout
        .iter()
        .position(|tile| tile.coordinate == state.robber)
        .ok_or(ImportError::InvalidRobber)? as u8;
    let mut ports = [None; 9];
    if state.ports.len() > ports.len() {
        return Err(ImportError::InvalidPort);
    }
    for (index, fixture) in state.ports.iter().enumerate() {
        ports[index] = Some(Port::new(
            fixture.resource.as_deref().map(resource_from).transpose()?,
            [
                NodeId::new(fixture.nodes[0]).map_err(|_| ImportError::InvalidPort)?,
                NodeId::new(fixture.nodes[1]).map_err(|_| ImportError::InvalidPort)?,
            ],
        ));
    }
    let context = GameContext::new(Layout::new(tiles).map_err(|_| ImportError::InvalidLayout)?)
        .with_ports(ports)
        .with_friendly_robber(state.friendly_robber);
    Ok((context, position))
}

fn wire_u8(value: &serde_json::Value) -> Result<u8, ImportError> {
    value
        .as_u64()
        .and_then(|raw| u8::try_from(raw).ok())
        .ok_or(ImportError::InvalidTrade)
}

fn bool_mask(values: &[bool]) -> u8 {
    values
        .iter()
        .enumerate()
        .fold(0, |mask, (index, value)| mask | (u8::from(*value) << index))
}

fn import_phase(state: &FixtureState, actor: PlayerId) -> Result<Phase, ImportError> {
    let (wire_actor, phase) = match &state.phase {
        FixturePhase::SetupSettlement { actor, reverse } => (
            actor,
            Phase::SetupSettlement {
                actor: seat(state, actor)?,
                reverse: *reverse,
            },
        ),
        FixturePhase::SetupRoad {
            actor,
            settlement,
            reverse,
        } => (
            actor,
            Phase::SetupRoad {
                actor: seat(state, actor)?,
                settlement: NodeId::new(*settlement).map_err(|_| ImportError::InvalidBuilding)?,
                reverse: *reverse,
            },
        ),
        FixturePhase::PreRoll { actor } => (
            actor,
            Phase::PreRoll {
                actor: seat(state, actor)?,
            },
        ),
        FixturePhase::PostRoll { actor } => (
            actor,
            Phase::PostRoll {
                actor: seat(state, actor)?,
            },
        ),
        FixturePhase::Discard { actor, remaining } => (
            actor,
            Phase::Discard {
                actor: seat(state, actor)?,
                remaining: *remaining,
            },
        ),
        FixturePhase::Robber {
            actor,
            resume_post_roll,
        } => (
            actor,
            Phase::Robber {
                actor: seat(state, actor)?,
                resume_post_roll: *resume_post_roll,
            },
        ),
        FixturePhase::FreeRoad {
            actor,
            remaining,
            resume_post_roll,
        } => (
            actor,
            Phase::FreeRoad {
                actor: seat(state, actor)?,
                remaining: *remaining,
                resume_post_roll: *resume_post_roll,
            },
        ),
        FixturePhase::TradeResponse { actor } => (
            actor,
            Phase::TradeResponse {
                actor: seat(state, actor)?,
            },
        ),
        FixturePhase::ChooseAccepter { actor } => (
            actor,
            Phase::ChooseAccepter {
                actor: seat(state, actor)?,
            },
        ),
    };
    if seat(state, wire_actor)? != actor {
        return Err(ImportError::InconsistentActor);
    }
    Ok(phase)
}

fn seat(state: &FixtureState, color: &str) -> Result<PlayerId, ImportError> {
    let index = state
        .colors
        .iter()
        .position(|candidate| candidate == color)
        .ok_or_else(|| ImportError::UnknownColor(color.to_owned()))?;
    PlayerId::new(index as u8).map_err(|_| ImportError::InvalidPlayerCount)
}

fn resource_from(value: &str) -> Result<Resource, ImportError> {
    match value {
        "WOOD" => Ok(Resource::Wood),
        "BRICK" => Ok(Resource::Brick),
        "SHEEP" => Ok(Resource::Sheep),
        "WHEAT" => Ok(Resource::Wheat),
        "ORE" => Ok(Resource::Ore),
        _ => Err(ImportError::UnknownResource(value.to_owned())),
    }
}

fn development(value: &str) -> Result<DevelopmentCard, ImportError> {
    match value {
        "KNIGHT" => Ok(DevelopmentCard::Knight),
        "YEAR_OF_PLENTY" => Ok(DevelopmentCard::YearOfPlenty),
        "MONOPOLY" => Ok(DevelopmentCard::Monopoly),
        "ROAD_BUILDING" => Ok(DevelopmentCard::RoadBuilding),
        "VICTORY_POINT" => Ok(DevelopmentCard::VictoryPoint),
        _ => Err(ImportError::UnknownDevelopmentCard(value.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_the_first_python_setup_fixture() {
        let line = include_str!("../../../tests/fixtures/transitions/sample-base-2p.jsonl")
            .lines()
            .next()
            .unwrap();
        let record: FixtureRecord = serde_json::from_str(line).unwrap();
        let (context, position) = import_state(&record.before).unwrap();
        assert_eq!(position.player_count, 2);
        assert!(matches!(position.phase, Phase::SetupSettlement { .. }));
        assert_eq!(
            context
                .layout
                .tile(catanatron_core::TileId::new(9).unwrap())
                .resource,
            None
        );
    }

    #[test]
    fn imports_every_committed_transition_boundary() {
        let corpora = [
            include_str!("../../../tests/fixtures/transitions/sample-base-2p.jsonl"),
            include_str!("../../../tests/fixtures/transitions/sample-base-3p.jsonl"),
            include_str!("../../../tests/fixtures/transitions/sample-base-4p.jsonl"),
            include_str!("../../../tests/fixtures/transitions/sample-tournament-4p.jsonl"),
            include_str!("../../../tests/fixtures/transitions/crafted-builds-and-trades.jsonl"),
        ];
        for corpus in corpora {
            for line in corpus.lines() {
                let record: FixtureRecord = serde_json::from_str(line).unwrap();
                import_state(&record.before)
                    .unwrap_or_else(|error| panic!("{} before: {error:?}", record.case_id));
                import_state(&record.after)
                    .unwrap_or_else(|error| panic!("{} after: {error:?}", record.case_id));
            }
        }
    }
}
