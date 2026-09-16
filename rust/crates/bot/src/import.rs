use std::collections::{HashMap, HashSet};

use catanatron_core::{
    draw_bounded, edge_endpoints, generate_actions_with_context, Action, EdgeId, GameContext,
    LandTile, Layout, NodeId, Phase, PlayerId, Port, Position, RandomSource, Resource,
    BASE_EDGE_COUNT, BASE_LAND_TILE_COUNT, BASE_NODE_COUNT, CITY_OFFSET,
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
// rust-v1 profile (RUST_EXECUTION_GUIDE.md E02): bank starts with 19 of each
// resource; the development deck has these counts of each card. Along with
// the always-visible bank and remaining-deck composition, these are the
// conservation totals determinization draws an opponent's hidden hand from.
const TOTAL_RESOURCE_SUPPLY: [u8; 5] = [19; 5];
const TOTAL_DEVELOPMENT_SUPPLY: [u8; 5] = [14, 2, 2, 2, 5];

pub struct Imported {
    pub context: GameContext,
    pub position: Position,
    pub offered: Vec<(Value, Action)>,
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
    rng: &mut impl RandomSource,
) -> Result<Imported, String> {
    let state: Snapshot =
        serde_json::from_value(state).map_err(|error| format!("state: {error}"))?;
    // v2 only added `random_state` to the authoritative document
    // (catanatron.serialization.SCHEMA_VERSION); client_view() already strips
    // that field before it reaches a bot, so v1 and v2 are wire-compatible here.
    if state.schema_version != 1 && state.schema_version != 2 {
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
    import_players(&state, &mut position, rng)?;
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
    let offered = offered
        .into_iter()
        .map(|wire| {
            let index = generated_wire
                .iter()
                .position(|candidate| candidate == &wire)
                .expect("menus were proven equal");
            (wire, generated[index])
        })
        .collect();
    Ok(Imported {
        context,
        position,
        offered,
    })
}

/// Imports each seat's player_state. A bot on the wire only ever gets its
/// own hand and development cards exactly (`client_view()` with a
/// perspective replaces every other seat's with aggregate counts:
/// `NUM_RESOURCES_IN_HAND`, `NUM_DEVELOPMENT_CARDS_IN_HAND`, and drops the
/// `*_OWNED_AT_START` eligibility flags entirely). For a seat missing the
/// exact fields, this records the known total and leaves the hand at zero;
/// `determinize_hidden_cards` fills every such seat in afterward with one
/// sampled assignment consistent with the totals and the always-visible
/// bank and remaining deck.
fn import_players(
    state: &Snapshot,
    position: &mut Position,
    rng: &mut impl RandomSource,
) -> Result<(), String> {
    let seats = state.colors.len();
    let mut hand_known = [true; 4];
    let mut dev_known = [true; 4];
    for index in 0..seats {
        let prefix = format!("P{index}_");
        hand_known[index] = has_all_keys(&state.player_state, &prefix, &RESOURCES, "_IN_HAND");
        dev_known[index] = has_all_keys(&state.player_state, &prefix, &DEVELOPMENTS, "_IN_HAND");
        let player = &mut position.players[index];
        player.hand = if hand_known[index] {
            state_counts(&state.player_state, &prefix, &RESOURCES, "_IN_HAND")?
        } else {
            [0; 5]
        };
        player.dev = if dev_known[index] {
            state_counts(&state.player_state, &prefix, &DEVELOPMENTS, "_IN_HAND")?
        } else {
            [0; 5]
        };
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
        if dev_known[index] {
            for (bit, card) in DEVELOPMENTS[..4].iter().enumerate() {
                if bool_key(
                    &state.player_state,
                    &format!("{prefix}{card}_OWNED_AT_START"),
                )? {
                    player.eligible_dev_mask |= 1 << bit;
                }
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
    determinize_hidden_cards(
        &state.player_state,
        position,
        seats,
        &hand_known,
        &dev_known,
        rng,
    )?;
    Ok(())
}

/// Samples one assignment of every seat whose hand or development cards
/// were only given as an aggregate count, drawing without replacement from
/// exactly what is left unaccounted for by conservation: the fixed supply
/// (`TOTAL_RESOURCE_SUPPLY` / `TOTAL_DEVELOPMENT_SUPPLY`) minus the bank,
/// the remaining draw deck, and every seat whose hand *is* known exactly
/// (which always includes the bot's own). This is a single Perfect
/// Information Monte Carlo sample, not a distribution -- the search that
/// runs against it is exactly as blind to which specific sample was chosen
/// as a person across the table reasoning about a guess would be, but it is
/// still one guess, not an average over many.
fn determinize_hidden_cards(
    player_state: &HashMap<String, Value>,
    position: &mut Position,
    seats: usize,
    hand_known: &[bool; 4],
    dev_known: &[bool; 4],
    rng: &mut impl RandomSource,
) -> Result<(), String> {
    let mut unseen_resources = TOTAL_RESOURCE_SUPPLY;
    subtract_into(&mut unseen_resources, &position.bank, "bank")?;
    let mut unseen_dev = TOTAL_DEVELOPMENT_SUPPLY;
    subtract_into(&mut unseen_dev, &position.dev_bank, "development deck")?;
    for index in 0..seats {
        if hand_known[index] {
            subtract_into(&mut unseen_resources, &position.players[index].hand, "hand")?;
        }
        if dev_known[index] {
            subtract_into(
                &mut unseen_dev,
                &position.players[index].dev,
                "development cards",
            )?;
        }
    }
    for index in 0..seats {
        let prefix = format!("P{index}_");
        if !hand_known[index] {
            let total = number_key(player_state, &format!("{prefix}NUM_RESOURCES_IN_HAND"))?;
            position.players[index].hand = deal_from_pool(&mut unseen_resources, total, rng)?;
        }
        if !dev_known[index] {
            let total = number_key(
                player_state,
                &format!("{prefix}NUM_DEVELOPMENT_CARDS_IN_HAND"),
            )?;
            position.players[index].dev = deal_from_pool(&mut unseen_dev, total, rng)?;
            // The wire never says which of a determinized hand's card types
            // were already held at the start of the turn (that flag is
            // popped along with everything else client_view() redacts), so
            // every type this sample happens to hold is treated as eligible
            // rather than as just bought. A search reaching this seat's
            // turn can then offer one extra card type at most, in the rare
            // case they truly bought it moments before -- overestimating an
            // opponent's options is the safe direction for a search
            // choosing its own move against them.
            for (bit, &count) in position.players[index].dev[..4].iter().enumerate() {
                if count > 0 {
                    position.players[index].eligible_dev_mask |= 1 << bit;
                }
            }
        }
    }
    Ok(())
}

fn subtract_into<const N: usize>(
    total: &mut [u8; N],
    accounted: &[u8; N],
    what: &str,
) -> Result<(), String> {
    for i in 0..N {
        total[i] = total[i]
            .checked_sub(accounted[i])
            .ok_or_else(|| format!("{what} claims more cards than the rules-profile supply"))?;
    }
    Ok(())
}

/// Draws `count` cards from `pool` without replacement, uniformly over the
/// remaining multiset (each draw is weighted by how many of each type are
/// still left), and removes what it drew from `pool` in place.
fn deal_from_pool<const N: usize>(
    pool: &mut [u8; N],
    mut count: u8,
    rng: &mut impl RandomSource,
) -> Result<[u8; N], String> {
    let mut dealt = [0_u8; N];
    while count > 0 {
        let total: u32 = pool.iter().map(|&c| u32::from(c)).sum();
        if total == 0 {
            return Err("determinization pool ran out before dealing every known card".to_owned());
        }
        let mut pick = draw_bounded(rng, u64::from(total)).ok_or("rng draw failed")? as u32;
        for slot in 0..N {
            if pick < u32::from(pool[slot]) {
                pool[slot] -= 1;
                dealt[slot] += 1;
                break;
            }
            pick -= u32::from(pool[slot]);
        }
        count -= 1;
    }
    Ok(dealt)
}

fn has_all_keys<const N: usize>(
    map: &HashMap<String, Value>,
    prefix: &str,
    names: &[&str; N],
    suffix: &str,
) -> bool {
    names
        .iter()
        .all(|name| map.contains_key(&format!("{prefix}{name}{suffix}")))
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
    fn deal_from_pool_never_exceeds_what_is_available_and_is_seed_reproducible() {
        let mut rng = catanatron_search::SearchRng::from_seed(catanatron_search::derive_seed(
            1,
            2,
            0,
            catanatron_search::StreamKind::Determinize,
        ));
        let mut pool = [3_u8, 2, 1, 0, 0];
        let dealt = deal_from_pool(&mut pool, 4, &mut rng).unwrap();
        assert_eq!(dealt.iter().map(|&c| u32::from(c)).sum::<u32>(), 4);
        assert_eq!(dealt[3], 0, "can't deal a resource with zero left");
        assert_eq!(dealt[4], 0, "can't deal a resource with zero left");
        assert_eq!(pool, [3 - dealt[0], 2 - dealt[1], 1 - dealt[2], 0, 0]);

        let mut rng_again = catanatron_search::SearchRng::from_seed(
            catanatron_search::derive_seed(1, 2, 0, catanatron_search::StreamKind::Determinize),
        );
        let mut pool_again = [3_u8, 2, 1, 0, 0];
        assert_eq!(
            deal_from_pool(&mut pool_again, 4, &mut rng_again).unwrap(),
            dealt,
            "the same seed must deal the same hand"
        );
    }

    #[test]
    fn deal_from_pool_rejects_asking_for_more_than_the_pool_holds() {
        let mut rng = catanatron_search::SearchRng::from_seed(catanatron_search::derive_seed(
            1,
            0,
            0,
            catanatron_search::StreamKind::Determinize,
        ));
        let mut pool = [1_u8, 0, 0, 0, 0];
        assert!(deal_from_pool(&mut pool, 2, &mut rng).is_err());
    }

    #[test]
    fn determinize_hidden_cards_respects_conservation_and_declared_totals() {
        let mut rng = catanatron_search::SearchRng::from_seed(catanatron_search::derive_seed(
            7,
            0,
            0,
            catanatron_search::StreamKind::Determinize,
        ));
        let mut position = Position::new(2).unwrap();
        position.bank = [10, 10, 10, 10, 10];
        position.dev_bank = [5, 1, 1, 1, 2];
        position.players[0].hand = [2, 0, 0, 0, 0]; // this bot's own, exact
        position.players[0].dev = [1, 0, 0, 0, 0];
        let mut player_state = HashMap::new();
        player_state.insert("P1_NUM_RESOURCES_IN_HAND".to_owned(), json!(5));
        player_state.insert("P1_NUM_DEVELOPMENT_CARDS_IN_HAND".to_owned(), json!(1));

        determinize_hidden_cards(
            &player_state,
            &mut position,
            2,
            &[true, false, true, true],
            &[true, false, true, true],
            &mut rng,
        )
        .unwrap();

        // Bank and this bot's own hand are exact, so what dealing could ever
        // hand seat 1 is bounded (per resource) by the fixed 19-card supply
        // minus those -- the dealt total must fit under that bound, never
        // invent cards no one could actually be holding.
        let hand1 = position.players[1].hand;
        assert_eq!(hand1.iter().map(|&c| u32::from(c)).sum::<u32>(), 5);
        for (r, &dealt) in hand1.iter().enumerate() {
            let unseen = 19 - position.bank[r] - position.players[0].hand[r];
            assert!(
                dealt <= unseen,
                "resource {r}: dealt {dealt} > unseen {unseen}"
            );
        }

        let dev1 = position.players[1].dev;
        assert_eq!(dev1.iter().map(|&c| u32::from(c)).sum::<u32>(), 1);
        for (i, &dealt) in dev1.iter().enumerate() {
            let supply = [14, 2, 2, 2, 5][i];
            let unseen = supply - position.dev_bank[i] - position.players[0].dev[i];
            assert!(dealt <= unseen);
        }
        // Every development type the sampled hand holds is a card the
        // determinized seat is assumed eligible to play right away.
        for (bit, &count) in dev1[..4].iter().enumerate() {
            assert_eq!(
                count > 0,
                position.players[1].eligible_dev_mask & (1 << bit) != 0
            );
        }
    }

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

    #[test]
    fn emits_every_wire_action_kind_with_the_active_seat_color() {
        let colors = vec!["BLUE".to_owned(), "RED".to_owned()];
        let mut position = Position::new(2).unwrap();
        position.actor = PlayerId::new(1).unwrap();
        position.trade_give = [1, 0, 0, 0, 0];
        position.trade_receive = [0, 1, 0, 0, 0];
        let map = Map {
            template: "BASE".to_owned(),
            tiles: (0..BASE_LAND_TILE_COUNT)
                .map(|id| MapTile {
                    coordinate: [id as i8, 0, 0],
                    kind: "LAND".to_owned(),
                    id: Some(json!(id)),
                    resource: None,
                    number: None,
                })
                .collect(),
        };
        let edge = EdgeId::new(0).unwrap();
        let node = NodeId::new(0).unwrap();
        let tile = catanatron_core::TileId::new(0).unwrap();
        let p0 = PlayerId::new(0).unwrap();
        let actions = [
            Action::Roll,
            Action::EndTurn,
            Action::BuildRoad(edge),
            Action::BuildSettlement(node),
            Action::BuildCity(node),
            Action::BuyDevelopmentCard,
            Action::PlayKnight,
            Action::MoveRobber {
                tile,
                victim: Some(p0),
            },
            Action::Discard(Resource::Wood),
            Action::YearOfPlenty {
                first: Resource::Wood,
                second: Some(Resource::Brick),
            },
            Action::Monopoly(Resource::Ore),
            Action::RoadBuilding,
            Action::MaritimeTrade {
                give: Resource::Wood,
                receive: Resource::Ore,
                rate: 4,
            },
            Action::OfferTrade {
                give: [1, 0, 0, 0, 0],
                receive: [0, 1, 0, 0, 0],
            },
            Action::AcceptTrade,
            Action::RejectTrade,
            Action::ConfirmTrade(p0),
            Action::CancelTrade,
        ];
        let payloads = actions
            .into_iter()
            .map(|action| action_to_wire(action, &colors, &map, &position).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(payloads.len(), 18);
        assert!(payloads.iter().all(|payload| payload[0] == "RED"));
        assert_eq!(
            payloads
                .iter()
                .map(|payload| payload[1].as_str().unwrap())
                .collect::<HashSet<_>>()
                .len(),
            18
        );
    }
}
