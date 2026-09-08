use std::{
    env,
    process::ExitCode,
    time::{Duration, Instant},
};

use catanatron_search::{
    initialize_base, play_agents, AgentConfig, AgentKind, NumberPlacement, RolloutLimits,
};
use serde_json::json;

fn main() -> ExitCode {
    match run() {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<serde_json::Value, String> {
    let mut args = env::args().skip(1);
    let name = args
        .next()
        .ok_or("usage: compare-agents POLICY GAMES [SIMULATIONS] [BUDGET_MS] [MAX_DEPTH]")?;
    let games: u64 = args
        .next()
        .ok_or("missing GAMES")?
        .parse()
        .map_err(|_| "invalid GAMES")?;
    let simulations: u32 = args
        .next()
        .unwrap_or_else(|| "10".to_owned())
        .parse()
        .map_err(|_| "invalid SIMULATIONS")?;
    let budget_ms: u64 = args
        .next()
        .unwrap_or_else(|| "20".to_owned())
        .parse()
        .map_err(|_| "invalid BUDGET_MS")?;
    let kind = parse_kind(&name)?;
    let max_depth: u8 = args
        .next()
        .unwrap_or_else(|| "2".to_owned())
        .parse()
        .map_err(|_| "invalid MAX_DEPTH")?;
    let contender = AgentConfig {
        kind,
        simulations,
        max_depth,
        budget: Duration::from_millis(budget_ms),
    };
    let random = AgentConfig::new(AgentKind::Random);
    let started = Instant::now();
    let mut wins = 0_u64;
    let mut truncations = 0_u64;
    let mut actions = 0_u64;
    for game in 0..games {
        let (context, root) = initialize_base(4, NumberPlacement::OfficialSpiral, 91, game)
            .map_err(|error| format!("{error:?}"))?;
        let seat = (game % 4) as usize;
        let mut agents = [random; 4];
        agents[seat] = contender;
        let result = play_agents(
            &context,
            &root,
            &agents,
            10_000 + game,
            RolloutLimits::default(),
        );
        wins += u64::from(
            result
                .winner
                .is_some_and(|winner| usize::from(winner.get()) == seat),
        );
        truncations += u64::from(result.truncation.is_some());
        actions += u64::from(result.player_actions);
    }
    let seconds = started.elapsed().as_secs_f64();
    Ok(json!({
        "engine": "rust",
        "policy": name,
        "games": games,
        "wins": wins,
        "win_rate": wins as f64 / games as f64,
        "truncations": truncations,
        "player_actions": actions,
        "seconds": seconds,
        "simulations": simulations,
        "budget_ms": budget_ms,
        "depth": max_depth,
        "seat_rotation": "game_index_mod_4",
        "opponents": "three Random players"
    }))
}

fn parse_kind(name: &str) -> Result<AgentKind, String> {
    match name {
        "simple" => Ok(AgentKind::Simple),
        "random" => Ok(AgentKind::Random),
        "weighted" => Ok(AgentKind::WeightedRandom),
        "victory" => Ok(AgentKind::VictoryPoint),
        "value" => Ok(AgentKind::ValueFunction),
        "playouts" => Ok(AgentKind::GreedyPlayouts),
        "alphabeta" => Ok(AgentKind::AlphaBeta),
        "same-turn" => Ok(AgentKind::SameTurnAlphaBeta),
        "mcts" => Ok(AgentKind::Mcts),
        _ => Err(format!("unknown policy: {name}")),
    }
}
