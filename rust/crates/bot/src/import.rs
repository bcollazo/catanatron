use std::collections::{HashMap, HashSet};

use catanatron_core::{
    edge_endpoints, generate_actions_with_context, Action, EdgeId, GameContext, LandTile, Layout,
    NodeId, Phase, PlayerId, Port, Position, Resource, BASE_EDGE_COUNT, BASE_LAND_TILE_COUNT,
    BASE_NODE_COUNT, CITY_OFFSET,
};
use serde::Deserialize;
use serde_json::{json, Value};

const PORT_NODES: [[u8; 2]; 9] = [
    [25, 26],
    [28, 29],
    [32, 33],
    [35, 36],
    [38, 39],
    [40, 44],
    [45, 47],
    [48, 49],
    [52, 53],
];
const RESOURCES: [&str; 5] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];
const DEVELOPMENTS: [&str; 5] = [
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
];

pub struct Imported {
    pub context: GameContext,
    pub position: Position,
    pub offered: Vec<Value>,
}

#[derive(Deserialize)]
struct Snapshot {
    schema_version: u64,
    game: Game,
    map: Map,
    board: Board,
    colors: Vec<String>,
    player_state: HashMap<String, Value>,
    buildings_by_color: HashMap<String, HashMap<String, Vec<Value>>>,
    resource_freqdeck: Vec<Value>,
    development_listdeck: HashMap<String, Value>,
    num_turns: Value,
    current_player_index: Value,
    current_turn_index: Value,
    current_prompt: String,
    is_initial_build_phase: bool,
    discard_counts: Vec<Value>,
    free_roads_available: Value,
    current_trade: Vec<Value>,
    acceptees: Vec<bool>,
}

#[derive(Deserialize)]
struct Game {
    id: String,
    friendly_robber: bool,
}

#[derive(Deserialize)]
struct Map {
    template: String,
    tiles: Vec<MapTile>,
}

#[derive(Deserialize)]
struct MapTile {
    coordinate: [i8; 3],
    #[serde(rename = "type")]
    kind: String,
    id: Option<Value>,
    resource: Option<String>,
    number: Option<Value>,
}

#[derive(Deserialize)]
struct Board {
    buildings: Vec<Value>,
    roads: Vec<Value>,
    robber_coordinate: [i8; 3],
    road_lengths: HashMap<String, Value>,
    road_color: Option<String>,
}

pub fn import(
    state: Value,
    game_id: &str,
    bot_color: &str,
    offered: Vec<Value>,
) -> Result<Imported, String> {
    let state: Snapshot =
        serde_json::from_value(state).map_err(|error| format!("state: {error}"))?;
    if state.schema_version != 1 {
        return Err(format!(
            "unsupported schema_version {}",
            state.schema_version
        ));
    }
    if state.game.id != game_id {
        return Err("state.game.id does not match message game_id".to_owned());
    }
    if state.map.template != "BASE" {
        return Err(format!("unsupported map template {:?}", state.map.template));
    }
    if !(2..=4).contains(&state.colors.len()) {
        return Err("colors must contain 2 to 4 seats".to_owned());
    }
    let unique: HashSet<&str> = state.colors.iter().map(String::as_str).collect();
    if unique.len() != state.colors.len() || !unique.contains(bot_color) {
        return Err("colors must be unique and include the bot color".to_owned());
    }

    let actor = indexed_player(
        &state.current_player_index,
        state.colors.len(),
        "current_player_index",
    )?;
    let owner = indexed_player(
        &state.current_turn_index,
        state.colors.len(),
        "current_turn_index",
    )?;
    let mut position =
        Position::new(state.colors.len() as u8).map_err(|e| format!("position: {e:?}"))?;
    position.actor = actor;
    position.turn_owner = owner;
    position.turns = u16_value(&state.num_turns, "num_turns")?;
    position.bank = counts(&state.resource_freqdeck, "resource_freqdeck")?;
    position.dev_bank = named_map_counts(&state.development_listdeck, &DEVELOPMENTS)?;
    import_players(&state, &mut position)?;
    import_board(&state, &mut position)?;
    let (context, robber) = import_map(
        &state.map,
        state.game.friendly_robber,
        state.board.robber_coordinate,
    )?;
    position.robber = robber;
    position.phase = import_phase(&state, actor, owner)?;
    import_trade(&state, &mut position)?;

    for action in &offered {
        validate_wire_action(action, &state.colors, actor)?;
    }
    let mut generated = Vec::new();
    generate_actions_with_context(&position, &context, &mut generated);
    let generated_wire = generated
        .iter()
        .map(|action| action_to_wire(*action, &state.colors, &state.map, &position))
        .collect::<Result<Vec<_>, _>>()?;
    if !same_values(&offered, &generated_wire) {
        return Err(format!(
            "unexplained root menu mismatch: host={} rust={}",
            Value::Array(offered.clone()),
            Value::Array(generated_wire)
        ));
    }
    Ok(Imported {
        context,
        position,
        offered,
    })
}

fn import_players(state: &Snapshot, position: &mut Position) -> Result<(), String> {
    for index in 0..state.colors.len() {
        let prefix = format!("P{index}_");
        let player = &mut position.players[index];
        player.hand = state_counts(&state.player_state, &prefix, &RESOURCES, "_IN_HAND")?;
        player.dev = state_counts(&state.player_state, &prefix, &DEVELOPMENTS, "_IN_HAND")?;
        player.pieces = state_counts(
            &state.player_state,
            &prefix,
            &["ROADS", "SETTLEMENTS", "CITIES"],
            "_AVAILABLE",
        )?;
        player.played_dev = bool_key(
            &state.player_state,
            &format!("{prefix}HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"),
        )?;
        player.played_knights = number_key(&state.player_state, &format!("{prefix}PLAYED_KNIGHT"))?;
        for (bit, card) in DEVELOPMENTS[..4].iter().enumerate() {
            if bool_key(
                &state.player_state,
                &format!("{prefix}{card}_OWNED_AT_START"),
            )? {
                player.eligible_dev_mask |= 1 << bit;
            }
        }
        position.longest_road_lengths[index] = state
            .board
            .road_lengths
            .get(&state.colors[index])
            .map(|value| u8_value(value, "board.road_lengths"))
            .transpose()?
            .unwrap_or(0);
        if bool_key(&state.player_state, &format!("{prefix}HAS_ROAD"))? {
            set_holder(&mut position.longest_road_holder, index, "longest road")?;
        }
        if bool_key(&state.player_state, &format!("{prefix}HAS_ARMY"))? {
            set_holder(&mut position.largest_army_holder, index, "largest army")?;
        }
    }
    if let Some(color) = &state.board.road_color {
        if position.longest_road_holder != Some(seat(&state.colors, color)?) {
            return Err("board.road_color disagrees with P<i>_HAS_ROAD".to_owned());
        }
    }
    Ok(())
}

fn import_board(state: &Snapshot, position: &mut Position) -> Result<(), String> {
    for building in &state.board.buildings {
        let fields = array(building, 3, "board.buildings")?;
        let node = usize::from(u8_value(&fields[0], "building node")?);
        if node >= BASE_NODE_COUNT || position.buildings[node] != 0 {
            return Err("invalid or duplicate building node".to_owned());
        }
        let owner = seat(&state.colors, string_value(&fields[1], "building color")?)?;
        position.buildings[node] = match string_value(&fields[2], "building type")? {
            "SETTLEMENT" => owner.get() + 1,
            "CITY" => owner.get() + 1 + CITY_OFFSET,
            _ => return Err("unknown building type".to_owned()),
        };
    }
    for road in &state.board.roads {
        let fields = array(road, 2, "board.roads")?;
        let ends = array(&fields[0], 2, "road endpoints")?;
        let edge = edge_from_nodes([
            u8_value(&ends[0], "road node")?,
            u8_value(&ends[1], "road node")?,
        ])?;
        let owner = seat(&state.colors, string_value(&fields[1], "road color")?)?.get() + 1;
        let slot = &mut position.roads[usize::from(edge.get())];
        if *slot != 0 && *slot != owner {
            return Err("duplicated road orientations disagree".to_owned());
        }
        *slot = owner;
    }
    Ok(())
}

fn import_map(
    map: &Map,
    friendly: bool,
    robber_coord: [i8; 3],
) -> Result<(GameContext, u8), String> {
    let mut land = [LandTile::DESERT; BASE_LAND_TILE_COUNT];
    let mut ports = [None; 9];
    let mut robber = None;
    let mut land_tiles: Vec<&MapTile> = map
        .tiles
        .iter()
        .filter(|tile| tile.kind == "LAND")
        .collect();
    land_tiles.sort_by_key(|tile| tile.coordinate);
    if land_tiles.len() != BASE_LAND_TILE_COUNT {
        return Err("map is missing land tiles".to_owned());
    }
    let mut source_ids = HashSet::new();
    for (id, tile) in land_tiles.into_iter().enumerate() {
        let source_id = u8_value(tile.id.as_ref().ok_or("land missing id")?, "land id")?;
        if !source_ids.insert(source_id) {
            return Err("duplicate Python land id".to_owned());
        }
        land[id] = match (&tile.resource, &tile.number) {
            (None, None) => LandTile::DESERT,
            (Some(resource_name), Some(number)) => {
                LandTile::producing(resource(resource_name)?, u8_value(number, "tile number")?)
            }
            _ => return Err("invalid land resource/number".to_owned()),
        };
        if tile.coordinate == robber_coord {
            robber = Some(id as u8);
        }
    }
    for tile in &map.tiles {
        match tile.kind.as_str() {
            "LAND" => {}
            "PORT" => {
                let id = usize::from(u8_value(
                    tile.id.as_ref().ok_or("port missing id")?,
                    "port id",
                )?);
                if id >= ports.len() || ports[id].is_some() {
                    return Err("invalid or duplicate port id".to_owned());
                }
                ports[id] = Some(Port::new(
                    tile.resource.as_deref().map(resource).transpose()?,
                    PORT_NODES[id].map(|node| NodeId::new(node).expect("static port node")),
                ));
            }
            "WATER" => {}
            _ => return Err("unknown map tile type".to_owned()),
        }
    }
    if ports.iter().any(Option::is_none) {
        return Err("map is missing ports".to_owned());
    }
    let layout = Layout::new(land).map_err(|error| format!("layout: {error:?}"))?;
    Ok((
        GameContext::new(layout)
            .with_ports(ports)
            .with_friendly_robber(friendly),
        robber.ok_or("robber is not on a land tile")?,
    ))
}

fn import_phase(state: &Snapshot, actor: PlayerId, owner: PlayerId) -> Result<Phase, String> {
    let rolled = bool_key(&state.player_state, &format!("P{}_HAS_ROLLED", owner.get()))?;
    if state.is_initial_build_phase {
        let built = state.board.buildings.len();
        return match state.current_prompt.as_str() {
            "BUILD_INITIAL_SETTLEMENT" => Ok(Phase::SetupSettlement {
                actor,
                reverse: built >= state.colors.len(),
            }),
            "BUILD_INITIAL_ROAD" => {
                let color = &state.colors[usize::from(actor.get())];
                let settlements = state
                    .buildings_by_color
                    .get(color)
                    .and_then(|items| items.get("SETTLEMENT"))
                    .ok_or("missing ordered setup settlements")?;
                let latest = settlements.last().ok_or("setup road has no settlement")?;
                Ok(Phase::SetupRoad {
                    actor,
                    settlement: NodeId::new(u8_value(latest, "latest settlement")?)
                        .map_err(|_| "invalid latest settlement")?,
                    reverse: built > state.colors.len(),
                })
            }
            _ => Err("initial build has invalid prompt".to_owned()),
        };
    }
    match state.current_prompt.as_str() {
        "DISCARD" => {
            if state.discard_counts.len() != state.colors.len() {
                return Err("discard_counts length mismatch".to_owned());
            }
            Ok(Phase::Discard {
                actor,
                remaining: u8_value(
                    &state.discard_counts[usize::from(actor.get())],
                    "discard count",
                )?,
            })
        }
        "MOVE_ROBBER" => Ok(Phase::Robber {
            actor,
            resume_post_roll: rolled,
        }),
        "DECIDE_TRADE" => Ok(Phase::TradeResponse { actor }),
        "DECIDE_ACCEPTEES" => Ok(Phase::ChooseAccepter { actor }),
        "PLAY_TURN" => {
            let roads = u8_value(&state.free_roads_available, "free_roads_available")?;
            if roads > 0 {
                Ok(Phase::FreeRoad {
                    actor,
                    remaining: roads,
                    resume_post_roll: rolled,
                })
            } else if rolled {
                Ok(Phase::PostRoll { actor })
            } else {
                Ok(Phase::PreRoll { actor })
            }
        }
        _ => Err("unknown current_prompt".to_owned()),
    }
}

fn import_trade(state: &Snapshot, position: &mut Position) -> Result<(), String> {
    if state.current_trade.len() != 11 || state.acceptees.len() != state.colors.len() {
        return Err("invalid trade state lengths".to_owned());
    }
    for index in 0..5 {
        position.trade_give[index] = u8_value(&state.current_trade[index], "trade give")?;
        position.trade_receive[index] = u8_value(&state.current_trade[index + 5], "trade receive")?;
    }
    position.trade_proposer = indexed_player(
        &state.current_trade[10],
        state.colors.len(),
        "trade proposer",
    )?;
    position.trade_accepted_mask = bool_mask(&state.acceptees);
    if matches!(position.phase, Phase::TradeResponse { .. }) {
        for index in 0..usize::from(position.actor.get()) {
            if index != usize::from(position.trade_proposer.get()) {
                position.trade_responded_mask |= 1 << index;
            }
        }
    } else if matches!(position.phase, Phase::ChooseAccepter { .. }) {
        position.trade_responded_mask =
            ((1_u8 << state.colors.len()) - 1) & !(1 << position.trade_proposer.get());
    }
    Ok(())
}

fn action_to_wire(
    action: Action,
    colors: &[String],
    map: &Map,
    position: &Position,
) -> Result<Value, String> {
    let color = &colors[usize::from(position.actor.get())];
    let value = match action {
        Action::Roll
        | Action::EndTurn
        | Action::BuyDevelopmentCard
        | Action::PlayKnight
        | Action::RoadBuilding
        | Action::AcceptTrade
        | Action::RejectTrade
        | Action::CancelTrade => Value::Null,
        Action::BuildRoad(edge) => {
            let (a, b) = edge_endpoints(edge);
            json!([a.get(), b.get()])
        }
        Action::BuildSettlement(node) | Action::BuildCity(node) => json!(node.get()),
        Action::MoveRobber { tile, victim } => {
            let mut land_tiles: Vec<&MapTile> = map
                .tiles
                .iter()
                .filter(|item| item.kind == "LAND")
                .collect();
            land_tiles.sort_by_key(|item| item.coordinate);
            let coordinate = land_tiles
                .get(usize::from(tile.get()))
                .ok_or("missing tile coordinate")?
                .coordinate;
            json!([
                coordinate,
                victim.map(|id| colors[usize::from(id.get())].clone())
            ])
        }
        Action::Discard(item) | Action::Monopoly(item) => json!(resource_name(item)),
        Action::YearOfPlenty { first, second } => match second {
            Some(second) => json!([resource_name(first), resource_name(second)]),
            None => json!([resource_name(first)]),
        },
        Action::MaritimeTrade {
            give,
            receive,
            rate,
        } => {
            let mut items = vec![Value::Null; 5];
            for item in items.iter_mut().take(usize::from(rate)) {
                *item = json!(resource_name(give));
            }
            items[4] = json!(resource_name(receive));
            Value::Array(items)
        }
        Action::OfferTrade { give, receive } => {
            Value::Array(give.into_iter().chain(receive).map(|v| json!(v)).collect())
        }
        Action::ConfirmTrade(player) => {
            let mut items: Vec<Value> = position
                .trade_give
                .into_iter()
                .chain(position.trade_receive)
                .map(|v| json!(v))
                .collect();
            items.push(json!(colors[usize::from(player.get())]));
            Value::Array(items)
        }
    };
    let kind = match action {
        Action::Roll => "ROLL",
        Action::EndTurn => "END_TURN",
        Action::BuildRoad(_) => "BUILD_ROAD",
        Action::BuildSettlement(_) => "BUILD_SETTLEMENT",
        Action::BuildCity(_) => "BUILD_CITY",
        Action::BuyDevelopmentCard => "BUY_DEVELOPMENT_CARD",
        Action::PlayKnight => "PLAY_KNIGHT_CARD",
        Action::MoveRobber { .. } => "MOVE_ROBBER",
        Action::Discard(_) => "DISCARD_RESOURCE",
        Action::YearOfPlenty { .. } => "PLAY_YEAR_OF_PLENTY",
        Action::Monopoly(_) => "PLAY_MONOPOLY",
        Action::RoadBuilding => "PLAY_ROAD_BUILDING",
        Action::MaritimeTrade { .. } => "MARITIME_TRADE",
        Action::OfferTrade { .. } => "OFFER_TRADE",
        Action::AcceptTrade => "ACCEPT_TRADE",
        Action::RejectTrade => "REJECT_TRADE",
        Action::ConfirmTrade(_) => "CONFIRM_TRADE",
        Action::CancelTrade => "CANCEL_TRADE",
    };
    Ok(json!([color, kind, value]))
}

fn validate_wire_action(value: &Value, colors: &[String], actor: PlayerId) -> Result<(), String> {
    let fields = array(value, 3, "offered action")?;
    if string_value(&fields[0], "action color")? != colors[usize::from(actor.get())]
        || !fields[1].is_string()
    {
        return Err("offered action has wrong actor or action type".to_owned());
    }
    Ok(())
}

fn same_values(left: &[Value], right: &[Value]) -> bool {
    left.len() == right.len()
        && left.iter().all(|value| right.contains(value))
        && right.iter().all(|value| left.contains(value))
}
fn edge_from_nodes(nodes: [u8; 2]) -> Result<EdgeId, String> {
    (0..BASE_EDGE_COUNT as u8)
        .filter_map(|raw| EdgeId::new(raw).ok())
        .find(|edge| {
            let (a, b) = edge_endpoints(*edge);
            [a.get(), b.get()] == nodes || [b.get(), a.get()] == nodes
        })
        .ok_or_else(|| format!("unknown edge {nodes:?}"))
}
fn state_counts<const N: usize>(
    map: &HashMap<String, Value>,
    prefix: &str,
    names: &[&str; N],
    suffix: &str,
) -> Result<[u8; N], String> {
    let mut out = [0; N];
    for (index, name) in names.iter().enumerate() {
        out[index] = number_key(map, &format!("{prefix}{name}{suffix}"))?;
    }
    Ok(out)
}
fn named_map_counts<const N: usize>(
    map: &HashMap<String, Value>,
    names: &[&str; N],
) -> Result<[u8; N], String> {
    let mut out = [0; N];
    for (index, name) in names.iter().enumerate() {
        out[index] = map
            .get(*name)
            .map(|v| u8_value(v, name))
            .transpose()?
            .unwrap_or(0);
    }
    Ok(out)
}
fn counts<const N: usize>(values: &[Value], field: &str) -> Result<[u8; N], String> {
    if values.len() != N {
        return Err(format!("{field} must have {N} counts"));
    }
    let mut out = [0; N];
    for (index, value) in values.iter().enumerate() {
        out[index] = u8_value(value, field)?;
    }
    Ok(out)
}
fn number_key(map: &HashMap<String, Value>, key: &str) -> Result<u8, String> {
    u8_value(
        map.get(key)
            .ok_or_else(|| format!("missing player_state.{key}"))?,
        key,
    )
}
fn bool_key(map: &HashMap<String, Value>, key: &str) -> Result<bool, String> {
    map.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{key} must be boolean"))
}
fn set_holder(slot: &mut Option<PlayerId>, index: usize, name: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("multiple {name} holders"));
    }
    *slot = PlayerId::new(index as u8).ok();
    Ok(())
}
fn indexed_player(value: &Value, count: usize, field: &str) -> Result<PlayerId, String> {
    let index = u8_value(value, field)?;
    if usize::from(index) >= count {
        return Err(format!("{field} out of range"));
    }
    PlayerId::new(index).map_err(|_| format!("{field} out of range"))
}
fn seat(colors: &[String], color: &str) -> Result<PlayerId, String> {
    colors
        .iter()
        .position(|item| item == color)
        .and_then(|index| PlayerId::new(index as u8).ok())
        .ok_or_else(|| format!("unknown color {color:?}"))
}
fn bool_mask(values: &[bool]) -> u8 {
    values
        .iter()
        .enumerate()
        .fold(0, |mask, (index, value)| mask | (u8::from(*value) << index))
}
fn resource(value: &str) -> Result<Resource, String> {
    match value {
        "WOOD" => Ok(Resource::Wood),
        "BRICK" => Ok(Resource::Brick),
        "SHEEP" => Ok(Resource::Sheep),
        "WHEAT" => Ok(Resource::Wheat),
        "ORE" => Ok(Resource::Ore),
        _ => Err(format!("unknown resource {value:?}")),
    }
}
fn resource_name(value: Resource) -> &'static str {
    match value {
        Resource::Wood => "WOOD",
        Resource::Brick => "BRICK",
        Resource::Sheep => "SHEEP",
        Resource::Wheat => "WHEAT",
        Resource::Ore => "ORE",
    }
}
fn array<'a>(value: &'a Value, length: usize, field: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .filter(|items| items.len() == length)
        .ok_or_else(|| format!("{field} must be an array of length {length}"))
}
fn string_value<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{field} must be a string"))
}
fn u8_value(value: &Value, field: &str) -> Result<u8, String> {
    value
        .as_u64()
        .and_then(|raw| u8::try_from(raw).ok())
        .ok_or_else(|| format!("{field} must be a u8"))
}
fn u16_value(value: &Value, field: &str) -> Result<u16, String> {
    value
        .as_u64()
        .and_then(|raw| u16::try_from(raw).ok())
        .ok_or_else(|| format!("{field} must be a u16"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_roads_with_different_owners() {
        assert_eq!(
            edge_from_nodes([0, 1]).unwrap(),
            edge_from_nodes([1, 0]).unwrap()
        );
    }

    #[test]
    fn rejects_numeric_overflow() {
        assert!(u8_value(&json!(256), "x").is_err());
        assert!(u16_value(&json!(-1), "x").is_err());
    }
}
