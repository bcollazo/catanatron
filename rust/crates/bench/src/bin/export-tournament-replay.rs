//! Plays a full TOURNAMENT game entirely inside the Rust engine (perfect
//! information, same as an in-process Python bot) with both seats using
//! `AgentKind::AlphaBeta`, and dumps the resulting (actor, action, outcome)
//! trace as JSON. `rust/tools/replay_trace_for_ui.py` force-replays this
//! trace through the real Python engine -- one committed action at a time,
//! via `apply_action(state, action, action_record)` -- and captures
//! `serialization.web_view()` after each ply, which is the document shape
//! ui-next's "Import JSON" already understands.
//!
//! TOURNAMENT is used (rather than BASE's shuffled layout) because both
//! engines build the exact same fixed tile/port assignment for it, so a node
//! or edge id chosen by Rust is legal in Python's independently-built board
//! without transferring any map geometry between the two.
use std::{env, fs, process::ExitCode, time::Duration};

use catanatron_core::{
    apply_checked_with_context, apply_outcome_checked_with_context, edge_endpoints_on, Action,
    DevelopmentCard, MapKind, Outcome, PlayerId, Resource, Status, Truncation,
};
use catanatron_search::{
    derive_seed, initialize_tournament, sample_outcome, select_agent_action, AgentConfig,
    AgentKind, SearchRng, StreamKind,
};
use serde_json::{json, Value};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut depth = 2_u8;
    let mut budget_ms = 200_u64;
    let mut seed = 42_u64;
    let mut turn_limit = 1_000_u16;
    let mut output = "rust/bench-results/tournament-ab-trace.json".to_owned();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--depth" => depth = value.parse().map_err(|_| "invalid --depth")?,
            "--budget-ms" => budget_ms = value.parse().map_err(|_| "invalid --budget-ms")?,
            "--seed" => seed = value.parse().map_err(|_| "invalid --seed")?,
            "--turn-limit" => turn_limit = value.parse().map_err(|_| "invalid --turn-limit")?,
            "--output" => output = value,
            _ => return Err(format!("unknown option {flag}")),
        }
    }

    let (context, root) =
        initialize_tournament(2).map_err(|error| format!("initialize_tournament: {error:?}"))?;
    let agent = AgentConfig {
        kind: AgentKind::AlphaBeta,
        simulations: 0,
        max_depth: depth,
        budget: Duration::from_millis(budget_ms),
    };
    let agents = [agent; 2];

    let mut position = root;
    let mut chance = SearchRng::from_seed(derive_seed(seed, 0, 0, StreamKind::Chance));
    let mut outcomes = Vec::with_capacity(36);
    let mut actions_buf = Vec::with_capacity(256);
    let mut trace = Vec::new();
    let mut player_actions: u64 = 0;
    let mut winner: Option<u8> = None;
    let mut truncation: Option<&'static str> = None;

    loop {
        if position.turns >= turn_limit {
            truncation = Some("turn_limit");
            break;
        }
        let actor = position.actor;
        let action = select_agent_action(
            &context,
            &position,
            agents[usize::from(actor.get())],
            seed.wrapping_add(player_actions),
            &mut actions_buf,
        )
        .ok_or("no legal action offered")?;
        let mut transition = apply_checked_with_context(&mut position, &context, actor, action)
            .map_err(|error| format!("apply {action:?}: {error:?}"))?;
        player_actions += 1;

        let mut outcome_json = Value::Null;
        if transition.status == Status::Chance {
            let outcome = sample_outcome(&position, &mut chance, &mut outcomes)
                .ok_or("chance phase offered no outcome")?;
            outcome_json = encode_outcome(outcome);
            transition = apply_outcome_checked_with_context(&mut position, &context, outcome)
                .map_err(|error| format!("apply outcome {outcome:?}: {error:?}"))?;
        }

        trace.push(json!({
            "actor": actor.get(),
            "action": encode_action(action),
            "outcome": outcome_json,
        }));

        match transition.status {
            Status::Won(color) => {
                winner = Some(color.get());
                break;
            }
            Status::Truncated(Truncation::TurnLimit) => {
                truncation = Some("turn_limit");
                break;
            }
            Status::Truncated(Truncation::ActionLimit) => {
                truncation = Some("action_limit");
                break;
            }
            Status::Decision | Status::Chance => {}
        }
    }

    let document = json!({
        "map": "TOURNAMENT",
        "player_count": 2,
        "seed": seed,
        "depth": depth,
        "winner": winner,
        "truncation": truncation,
        "turns": position.turns,
        "player_actions": player_actions,
        "trace": trace,
    });
    eprintln!(
        "played {player_actions} actions over {} turns; winner={winner:?} truncation={truncation:?}",
        position.turns
    );
    fs::write(
        &output,
        serde_json::to_string(&document).map_err(|e| e.to_string())?,
    )
    .map_err(|error| format!("writing {output}: {error}"))?;
    eprintln!("wrote {output}");
    Ok(())
}

fn encode_action(action: Action) -> Value {
    match action {
        Action::Roll => json!({"type": "ROLL"}),
        Action::EndTurn => json!({"type": "END_TURN"}),
        Action::BuildRoad(edge) => {
            let (a, b) = edge_endpoints_on(MapKind::Base, edge).expect("active edge");
            json!({"type": "BUILD_ROAD", "value": [a.get(), b.get()]})
        }
        Action::BuildSettlement(node) => {
            json!({"type": "BUILD_SETTLEMENT", "value": node.get()})
        }
        Action::BuildCity(node) => json!({"type": "BUILD_CITY", "value": node.get()}),
        Action::BuyDevelopmentCard => json!({"type": "BUY_DEVELOPMENT_CARD"}),
        Action::PlayKnight => json!({"type": "PLAY_KNIGHT_CARD"}),
        Action::MoveRobber { tile, victim } => json!({
            "type": "MOVE_ROBBER",
            "value": [tile.get(), victim.map(PlayerId::get)],
        }),
        Action::Discard(resource) => {
            json!({"type": "DISCARD_RESOURCE", "value": encode_resource(resource)})
        }
        Action::YearOfPlenty { first, second } => json!({
            "type": "PLAY_YEAR_OF_PLENTY",
            "value": [
                encode_resource(first),
                second.map(encode_resource),
            ],
        }),
        Action::Monopoly(resource) => {
            json!({"type": "PLAY_MONOPOLY", "value": encode_resource(resource)})
        }
        Action::RoadBuilding => json!({"type": "PLAY_ROAD_BUILDING"}),
        Action::MaritimeTrade {
            give,
            receive,
            rate,
        } => json!({
            "type": "MARITIME_TRADE",
            "value": {"give": encode_resource(give), "receive": encode_resource(receive), "rate": rate},
        }),
        Action::OfferTrade { .. }
        | Action::AcceptTrade
        | Action::RejectTrade
        | Action::ConfirmTrade(_)
        | Action::CancelTrade => json!({"type": "UNSUPPORTED_TRADE"}),
    }
}

fn encode_outcome(outcome: Outcome) -> Value {
    match outcome {
        Outcome::Dice { first, second } => json!({"type": "DICE", "value": [first, second]}),
        Outcome::StolenResource(resource) => {
            json!({"type": "STOLEN_RESOURCE", "value": encode_resource(resource)})
        }
        Outcome::DevelopmentCard(card) => {
            json!({"type": "DEVELOPMENT_CARD", "value": encode_development(card)})
        }
    }
}

fn encode_resource(resource: Resource) -> &'static str {
    match resource {
        Resource::Wood => "WOOD",
        Resource::Brick => "BRICK",
        Resource::Sheep => "SHEEP",
        Resource::Wheat => "WHEAT",
        Resource::Ore => "ORE",
    }
}

fn encode_development(card: DevelopmentCard) -> &'static str {
    match card {
        DevelopmentCard::Knight => "KNIGHT",
        DevelopmentCard::YearOfPlenty => "YEAR_OF_PLENTY",
        DevelopmentCard::Monopoly => "MONOPOLY",
        DevelopmentCard::RoadBuilding => "ROAD_BUILDING",
        DevelopmentCard::VictoryPoint => "VICTORY_POINT",
    }
}
