use catanatron_core::{
    apply_checked_with_context, apply_outcome_checked_with_context, edge_endpoints,
    generate_actions_with_context, Action, DevelopmentCard, EdgeId, GameContext, LandTile, Layout,
    NodeId, Outcome, Phase, PlayerId, Port, Position, Resource, Status, TileId, BASE_EDGE_COUNT,
    BASE_LAND_TILE_COUNT, BASE_NODE_COUNT,
};
use serde::{Deserialize, Serialize};

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
    Terminal {
        winner: String,
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
    InvalidAction(String),
    InvalidOutcome,
}

#[derive(Debug, Serialize)]
pub struct ConformanceError {
    pub case_id: String,
    pub field: String,
    pub expected: String,
    pub actual: String,
}

pub fn check_record(record: &FixtureRecord) -> Result<(), ConformanceError> {
    let (context, mut actual) = import_state(&record.before)
        .map_err(|error| mismatch(record, "before", "importable", format!("{error:?}")))?;
    compare_menu_boundary(
        record,
        "legal_before",
        &context,
        &actual,
        &record.before,
        &record.legal_before,
    )?;
    let before_position = actual;
    let intent = wire_action(&record.before, &record.action)
        .map_err(|error| mismatch(record, "action", "valid", format!("{error:?}")))?;
    let actor = actual.actor;
    let mut transition = apply_checked_with_context(&mut actual, &context, actor, intent)
        .map_err(|error| mismatch(record, "apply", "accepted", format!("{error:?}")))?;
    if matches!(
        record.action.kind.as_str(),
        "ROLL" | "BUY_DEVELOPMENT_CARD" | "MOVE_ROBBER"
    ) {
        if let Some(value) = &record.outcome {
            let outcome = wire_outcome(&record.action.kind, value)
                .map_err(|error| mismatch(record, "outcome", "valid", format!("{error:?}")))?;
            transition = apply_outcome_checked_with_context(&mut actual, &context, outcome)
                .map_err(|error| {
                    mismatch(record, "outcome.apply", "accepted", format!("{error:?}"))
                })?;
        }
    }
    let (_, expected) = import_state(&record.after)
        .map_err(|error| mismatch(record, "after", "importable", format!("{error:?}")))?;
    if actual != expected {
        let field = longest_road_divergence(&expected, &actual)
            .or_else(|| incumbent_tie_divergence(&before_position, &expected, &actual))
            .or_else(|| below_threshold_award_divergence(&expected, &actual))
            .unwrap_or_else(|| first_position_difference(&expected, &actual));
        return Err(mismatch(
            record,
            field,
            format!("{expected:?}"),
            format!("{actual:?}"),
        ));
    }
    let expected_status = match record.status_after.as_str() {
        "decision" => Status::Decision,
        "won" => match &record.after.phase {
            FixturePhase::Terminal { winner } => Status::Won(
                seat(&record.after, winner)
                    .map_err(|error| mismatch(record, "winner", "valid", format!("{error:?}")))?,
            ),
            _ => return Err(mismatch(record, "after.phase", "terminal", "non-terminal")),
        },
        other => {
            return Err(mismatch(
                record,
                "status_after",
                "decision",
                other.to_owned(),
            ))
        }
    };
    if transition.status != expected_status {
        return Err(mismatch(
            record,
            "status_after",
            format!("{expected_status:?}"),
            format!("{:?}", transition.status),
        ));
    }
    compare_menu_boundary(
        record,
        "legal_after",
        &context,
        &actual,
        &record.after,
        &record.legal_after,
    )
}

fn below_threshold_award_divergence(
    expected: &Position,
    actual: &Position,
) -> Option<&'static str> {
    let python_holder = expected.longest_road_holder?;
    if actual.longest_road_holder.is_some()
        || expected.longest_road_lengths != actual.longest_road_lengths
        || expected.longest_road_lengths[usize::from(python_holder.get())] >= 5
    {
        return None;
    }
    let mut normalized = *expected;
    normalized.longest_road_holder = None;
    normalize_terminal_consequence(&mut normalized, actual);
    (normalized == *actual).then_some("divergence:D005-longest-road-below-threshold-award")
}

fn incumbent_tie_divergence(
    before: &Position,
    expected: &Position,
    actual: &Position,
) -> Option<&'static str> {
    let incumbent = before.longest_road_holder?;
    if actual.longest_road_holder != Some(incumbent)
        || expected.longest_road_holder == actual.longest_road_holder
        || expected.longest_road_lengths != actual.longest_road_lengths
    {
        return None;
    }
    let expected_holder = expected.longest_road_holder?;
    let maximum = actual.longest_road_lengths[usize::from(incumbent.get())];
    if maximum < 5 || actual.longest_road_lengths[usize::from(expected_holder.get())] != maximum {
        return None;
    }
    let mut normalized = *expected;
    normalized.longest_road_holder = actual.longest_road_holder;
    normalize_terminal_consequence(&mut normalized, actual);
    (normalized == *actual).then_some("divergence:D004-longest-road-incumbent-tie-retention")
}

fn longest_road_divergence(expected: &Position, actual: &Position) -> Option<&'static str> {
    let mut normalized = *expected;
    normalized.longest_road_lengths = actual.longest_road_lengths;
    normalized.longest_road_holder = actual.longest_road_holder;
    normalize_terminal_consequence(&mut normalized, actual);
    if normalized != *actual {
        return None;
    }
    let affected = (0..expected.player_count).find(|raw| {
        actual.longest_road_lengths[usize::from(*raw)]
            > expected.longest_road_lengths[usize::from(*raw)]
    })?;
    let player = PlayerId::new(affected).expect("active seat");
    let enters_opponent = expected.roads.iter().enumerate().any(|(index, owner)| {
        if *owner != affected + 1 {
            return false;
        }
        let edge = EdgeId::new(index as u8).expect("base edge");
        let (a, b) = edge_endpoints(edge);
        [a, b].iter().any(|node| {
            catanatron_core::building_owner(expected.buildings[usize::from(node.get())])
                .is_some_and(|building_owner| building_owner != player)
        })
    });
    Some(if enters_opponent {
        "divergence:D002-longest-road-entering-opponent-building"
    } else {
        "divergence:D003-longest-road-branch-undercount"
    })
}

fn normalize_terminal_consequence(expected: &mut Position, actual: &Position) {
    if matches!(expected.phase, Phase::Terminal) && !matches!(actual.phase, Phase::Terminal) {
        expected.phase = actual.phase;
    }
}

fn mismatch(
    record: &FixtureRecord,
    field: impl Into<String>,
    expected: impl Into<String>,
    actual: impl Into<String>,
) -> ConformanceError {
    ConformanceError {
        case_id: record.case_id.clone(),
        field: field.into(),
        expected: expected.into(),
        actual: actual.into(),
    }
}

fn compare_menu_boundary(
    record: &FixtureRecord,
    field: &str,
    context: &GameContext,
    position: &Position,
    state: &FixtureState,
    wire_menu: &[WireAction],
) -> Result<(), ConformanceError> {
    let mut actual = Vec::new();
    generate_actions_with_context(position, context, &mut actual);
    let expected = wire_menu
        .iter()
        .map(|action| wire_action(state, action))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| mismatch(record, field, "valid", format!("{error:?}")))?;
    if actual.len() != expected.len()
        || actual.iter().any(|action| !expected.contains(action))
        || expected.iter().any(|action| !actual.contains(action))
    {
        return Err(mismatch(
            record,
            field,
            format!("{expected:?}"),
            format!("{actual:?}"),
        ));
    }
    Ok(())
}

fn first_position_difference(expected: &Position, actual: &Position) -> &'static str {
    if expected.players != actual.players {
        "after.players"
    } else if expected.buildings != actual.buildings {
        "after.buildings"
    } else if expected.roads != actual.roads {
        "after.roads"
    } else if expected.bank != actual.bank {
        "after.bank"
    } else if expected.dev_bank != actual.dev_bank {
        "after.development_deck"
    } else if expected.robber != actual.robber {
        "after.robber"
    } else if expected.phase != actual.phase {
        "after.phase"
    } else if expected.actor != actual.actor {
        "after.actor"
    } else if expected.turn_owner != actual.turn_owner {
        "after.turn_owner"
    } else if expected.turns != actual.turns {
        "after.turns"
    } else if expected.trade_give != actual.trade_give
        || expected.trade_receive != actual.trade_receive
        || expected.trade_proposer != actual.trade_proposer
        || expected.trade_responded_mask != actual.trade_responded_mask
        || expected.trade_accepted_mask != actual.trade_accepted_mask
    {
        "after.trade"
    } else {
        "after.awards"
    }
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
        FixturePhase::Terminal { .. } => (&state.actor, Phase::Terminal),
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

fn wire_action(state: &FixtureState, wire: &WireAction) -> Result<Action, ImportError> {
    if seat(state, &wire.color)? != state_actor(state)? {
        return Err(ImportError::InconsistentActor);
    }
    let invalid = || ImportError::InvalidAction(wire.kind.clone());
    Ok(match wire.kind.as_str() {
        "ROLL" => Action::Roll,
        "END_TURN" => Action::EndTurn,
        "BUILD_ROAD" => Action::BuildRoad(edge_from_value(&wire.value)?),
        "BUILD_SETTLEMENT" => Action::BuildSettlement(
            NodeId::new(json_u8(&wire.value, invalid())?).map_err(|_| invalid())?,
        ),
        "BUILD_CITY" => {
            Action::BuildCity(NodeId::new(json_u8(&wire.value, invalid())?).map_err(|_| invalid())?)
        }
        "BUY_DEVELOPMENT_CARD" => Action::BuyDevelopmentCard,
        "PLAY_KNIGHT_CARD" => Action::PlayKnight,
        "MOVE_ROBBER" => {
            let values = wire.value.as_array().ok_or_else(invalid)?;
            if values.len() != 2 {
                return Err(invalid());
            }
            let coordinate: [i8; 3] =
                serde_json::from_value(values[0].clone()).map_err(|_| invalid())?;
            let tile = state
                .layout
                .iter()
                .position(|candidate| candidate.coordinate == coordinate)
                .and_then(|index| TileId::new(index as u8).ok())
                .ok_or_else(invalid)?;
            let victim = values[1]
                .as_str()
                .map(|color| seat(state, color))
                .transpose()?;
            Action::MoveRobber { tile, victim }
        }
        "DISCARD_RESOURCE" => {
            Action::Discard(resource_from(wire.value.as_str().ok_or_else(invalid)?)?)
        }
        "PLAY_YEAR_OF_PLENTY" => {
            let values = wire.value.as_array().ok_or_else(invalid)?;
            let first = resource_from(
                values
                    .first()
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(invalid)?,
            )?;
            let second = values
                .get(1)
                .and_then(serde_json::Value::as_str)
                .map(resource_from)
                .transpose()?;
            Action::YearOfPlenty { first, second }
        }
        "PLAY_MONOPOLY" => {
            Action::Monopoly(resource_from(wire.value.as_str().ok_or_else(invalid)?)?)
        }
        "PLAY_ROAD_BUILDING" => Action::RoadBuilding,
        "MARITIME_TRADE" => {
            let values = wire.value.as_array().ok_or_else(invalid)?;
            let mut counts = [0_u8; 5];
            for value in values {
                if let Some(resource) = value.as_str() {
                    counts[resource_from(resource)?.index()] += 1;
                } else if !value.is_null() {
                    return Err(invalid());
                }
            }
            let give = Resource::ALL
                .into_iter()
                .find(|resource| counts[resource.index()] > 1)
                .ok_or_else(invalid)?;
            let receive = Resource::ALL
                .into_iter()
                .find(|resource| counts[resource.index()] == 1)
                .ok_or_else(invalid)?;
            Action::MaritimeTrade {
                give,
                receive,
                rate: counts[give.index()],
            }
        }
        "OFFER_TRADE" => {
            let (give, receive) = trade_arrays(&wire.value)?;
            Action::OfferTrade { give, receive }
        }
        "ACCEPT_TRADE" => Action::AcceptTrade,
        "REJECT_TRADE" => Action::RejectTrade,
        "CONFIRM_TRADE" => {
            let values = wire.value.as_array().ok_or_else(invalid)?;
            let color = values
                .get(10)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(invalid)?;
            Action::ConfirmTrade(seat(state, color)?)
        }
        "CANCEL_TRADE" => Action::CancelTrade,
        _ => return Err(invalid()),
    })
}

fn state_actor(state: &FixtureState) -> Result<PlayerId, ImportError> {
    seat(state, &state.actor)
}

fn json_u8(value: &serde_json::Value, error: ImportError) -> Result<u8, ImportError> {
    value
        .as_u64()
        .and_then(|raw| u8::try_from(raw).ok())
        .ok_or(error)
}

fn edge_from_value(value: &serde_json::Value) -> Result<EdgeId, ImportError> {
    let nodes: [u8; 2] = serde_json::from_value(value.clone())
        .map_err(|_| ImportError::InvalidAction("BUILD_ROAD".to_owned()))?;
    (0..BASE_EDGE_COUNT as u8)
        .map(|raw| EdgeId::new(raw).expect("base edge"))
        .find(|edge| {
            let (a, b) = edge_endpoints(*edge);
            [a.get(), b.get()] == nodes || [b.get(), a.get()] == nodes
        })
        .ok_or(ImportError::InvalidRoad(nodes))
}

fn trade_arrays(value: &serde_json::Value) -> Result<([u8; 5], [u8; 5]), ImportError> {
    let values = value
        .as_array()
        .filter(|values| values.len() >= 10)
        .ok_or(ImportError::InvalidTrade)?;
    let mut give = [0; 5];
    let mut receive = [0; 5];
    for index in 0..5 {
        give[index] = wire_u8(&values[index])?;
        receive[index] = wire_u8(&values[index + 5])?;
    }
    Ok((give, receive))
}

fn wire_outcome(kind: &str, value: &serde_json::Value) -> Result<Outcome, ImportError> {
    match kind {
        "ROLL" => {
            let dice: [u8; 2] =
                serde_json::from_value(value.clone()).map_err(|_| ImportError::InvalidOutcome)?;
            Ok(Outcome::Dice {
                first: dice[0],
                second: dice[1],
            })
        }
        "BUY_DEVELOPMENT_CARD" => Ok(Outcome::DevelopmentCard(development(
            value.as_str().ok_or(ImportError::InvalidOutcome)?,
        )?)),
        "MOVE_ROBBER" => Ok(Outcome::StolenResource(resource_from(
            value.as_str().ok_or(ImportError::InvalidOutcome)?,
        )?)),
        _ => Err(ImportError::InvalidOutcome),
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

    #[test]
    fn applies_every_committed_python_transition() {
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
                if let Err(error) = check_record(&record) {
                    panic!("{error:#?}");
                }
            }
        }
    }
}
