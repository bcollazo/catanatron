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
    let requested_games = parse(&mut args, "GAMES", 100)?;
    let budget_ms = parse(&mut args, "BUDGET_MS", 20)?;
    let wall_seconds = parse(&mut args, "WALL_SECONDS", 3_000)?;
    let depths_arg = args.next().unwrap_or_else(|| "1,2,3,4".to_owned());
    let depths = depths_arg
        .split(',')
        .map(|value| value.parse::<u8>().map_err(|_| "invalid DEPTHS".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    if depths.len() != 4
        || depths.contains(&0)
        || (0..depths.len()).any(|index| depths[index + 1..].contains(&depths[index]))
    {
        return Err("DEPTHS must contain four distinct positive comma-separated depths".to_owned());
    }
    let started = Instant::now();
    let wall_limit = Duration::from_secs(wall_seconds);
    let mut wins = vec![0_u64; 4];
    let mut seats = [[0_u64; 4]; 4];
    let mut truncations = 0_u64;
    let mut completed_games = 0_u64;

    for game_index in 0..requested_games {
        if started.elapsed() >= wall_limit {
            break;
        }
        let block = game_index / 4;
        let rotation = (game_index % 4) as usize;
        let (context, root) = initialize_base(4, NumberPlacement::OfficialSpiral, 91, block)
            .map_err(|error| format!("{error:?}"))?;
        let mut agents = Vec::with_capacity(4);
        let mut depth_at_seat = [0_u8; 4];
        for seat in 0..4 {
            let depth = depths[(seat + rotation) % 4];
            depth_at_seat[seat] = depth;
            agents.push(AgentConfig {
                kind: AgentKind::AlphaBeta,
                simulations: 0,
                max_depth: depth,
                budget: Duration::from_millis(budget_ms),
            });
        }
        let result = play_agents(
            &context,
            &root,
            &agents,
            10_000 + block,
            RolloutLimits::default(),
        );
        completed_games += 1;
        if let Some(winner) = result.winner {
            let seat = usize::from(winner.get());
            let depth_index = depths
                .iter()
                .position(|&depth| depth == depth_at_seat[seat])
                .expect("assigned depth");
            wins[depth_index] += 1;
            seats[depth_index][seat] += 1;
        } else {
            truncations += 1;
        }
    }

    Ok(json!({
        "engine": "rust",
        "depths": depths,
        "budget_ms_per_decision": budget_ms,
        "requested_games": requested_games,
        "completed_games": completed_games,
        "truncations": truncations,
        "wins": wins,
        "win_rates": wins.iter().map(|&count| count as f64 / completed_games as f64).collect::<Vec<_>>(),
        "wins_by_depth_and_seat": seats,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "wall_limit_seconds": wall_seconds,
        "design": "four-player blocks; depths rotate through every seat on the same board and RNG seed",
    }))
}

fn parse<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
    default: T,
) -> Result<T, String> {
    match args.next() {
        Some(value) => value.parse().map_err(|_| format!("invalid {name}")),
        None => Ok(default),
    }
}
