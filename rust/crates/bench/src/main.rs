use std::{env, fs, process::ExitCode, time::Instant};

use catanatron_core::{generate_actions_with_context, Position};
use catanatron_search::{
    initialize_base, rollout, NumberPlacement, Policy, RolloutLimits, RolloutScratch,
};
use serde_json::{json, Value};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};

#[global_allocator]
static GLOBAL: &StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

#[derive(Clone, Debug)]
struct Args {
    command: String,
    seed: u64,
    players: u8,
    policy: Policy,
    games: u64,
    rollouts: u64,
    fixtures: u64,
    output: Option<String>,
}

fn main() -> ExitCode {
    match parse().and_then(run) {
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

fn parse() -> Result<Args, String> {
    let mut values = env::args().skip(1);
    let command = values.next().ok_or_else(usage)?;
    if !matches!(
        command.as_str(),
        "games" | "rollouts" | "kernels" | "allocations"
    ) {
        return Err(usage());
    }
    let mut args = Args {
        command,
        seed: 8_600,
        players: 4,
        policy: Policy::Weighted,
        games: 100,
        rollouts: 1_000,
        fixtures: 1,
        output: None,
    };
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--seed" => args.seed = number(&flag, &value)?,
            "--players" => args.players = number(&flag, &value)?,
            "--policy" => {
                args.policy = match value.as_str() {
                    "random" => Policy::Random,
                    "weighted" => Policy::Weighted,
                    _ => return Err("--policy must be random or weighted".to_owned()),
                }
            }
            "--games" => args.games = number(&flag, &value)?,
            "--rollouts" => args.rollouts = number(&flag, &value)?,
            "--fixtures" => args.fixtures = number(&flag, &value)?,
            "--output" => args.output = Some(value),
            "--map" if value == "BASE" => {}
            "--threads" if value == "1" => {}
            "--map" => return Err("only BASE is implemented before E12 MINI support".to_owned()),
            "--threads" => return Err("this baseline command supports --threads 1".to_owned()),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if !(2..=4).contains(&args.players)
        || args.games == 0
        || args.rollouts == 0
        || args.fixtures != 1
    {
        return Err(
            "players must be 2..=4, counts positive, and baseline --fixtures must be 1".to_owned(),
        );
    }
    Ok(args)
}

fn number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn usage() -> String {
    "usage: catanatron-bench <games|rollouts|kernels|allocations> [--seed N] [--players 2..4] [--policy random|weighted] [--games N] [--rollouts N] [--threads 1] [--map BASE] [--output PATH]".to_owned()
}

fn run(args: Args) -> Result<Value, String> {
    let report = match args.command.as_str() {
        "games" => timed_rollouts(&args, args.games, "games"),
        "rollouts" => timed_rollouts(&args, args.rollouts, "rollouts"),
        "kernels" => kernels(&args),
        "allocations" => allocations(&args),
        _ => unreachable!(),
    }?;
    if let Some(path) = &args.output {
        fs::write(path, serde_json::to_string_pretty(&report).unwrap() + "\n")
            .map_err(|error| format!("{path}: {error}"))?;
    }
    Ok(report)
}

fn timed_rollouts(args: &Args, count: u64, workload: &str) -> Result<Value, String> {
    let mut samples = Vec::new();
    let mut total_actions = 0_u64;
    let mut completed = 0_u64;
    let mut truncated = 0_u64;
    let fixed = if workload == "rollouts" {
        Some(
            initialize_base(args.players, NumberPlacement::OfficialSpiral, args.seed, 0)
                .map_err(|error| format!("initialization failed: {error:?}"))?,
        )
    } else {
        None
    };
    for batch in 0..5 {
        let started = Instant::now();
        let mut scratch = RolloutScratch::default();
        for index in 0..count {
            let game_index = batch * count + index;
            let (context, root) = match fixed {
                Some(pair) => pair,
                None => initialize_base(
                    args.players,
                    NumberPlacement::OfficialSpiral,
                    args.seed,
                    game_index,
                )
                .map_err(|error| format!("initialization failed: {error:?}"))?,
            };
            let result = rollout(
                &context,
                &root,
                args.policy,
                args.seed.wrapping_add(game_index),
                RolloutLimits::default(),
                &mut scratch,
            );
            total_actions += u64::from(result.player_actions);
            if result.winner.is_some() {
                completed += 1;
            } else {
                truncated += 1;
            }
        }
        samples.push(started.elapsed().as_secs_f64());
    }
    let seconds: f64 = samples.iter().sum();
    Ok(common_report(
        args,
        workload,
        samples,
        json!({
            "games": count * 5,
            "completed": completed,
            "truncated": truncated,
            "player_intents": total_actions,
            "intents_per_second": total_actions as f64 / seconds,
            "state_bytes": std::mem::size_of::<Position>(),
        }),
    ))
}

fn kernels(args: &Args) -> Result<Value, String> {
    let (context, position) =
        initialize_base(args.players, NumberPlacement::OfficialSpiral, args.seed, 0)
            .map_err(|error| format!("initialization failed: {error:?}"))?;
    let mut actions = Vec::with_capacity(256);
    let mut samples = Vec::new();
    let iterations = args.rollouts;
    for _ in 0..5 {
        let started = Instant::now();
        for _ in 0..iterations {
            generate_actions_with_context(&position, &context, &mut actions);
            std::hint::black_box(&actions);
        }
        samples.push(started.elapsed().as_secs_f64());
    }
    let seconds: f64 = samples.iter().sum();
    Ok(common_report(
        args,
        "kernels",
        samples,
        json!({
            "iterations": iterations * 5,
            "iterations_per_second": iterations as f64 * 5.0 / seconds,
            "menu_capacity": actions.capacity(),
        }),
    ))
}

fn allocations(args: &Args) -> Result<Value, String> {
    let (context, root) =
        initialize_base(args.players, NumberPlacement::OfficialSpiral, args.seed, 0)
            .map_err(|error| format!("initialization failed: {error:?}"))?;
    let mut scratch = RolloutScratch::default();
    let _ = rollout(
        &context,
        &root,
        args.policy,
        args.seed,
        RolloutLimits::default(),
        &mut scratch,
    );
    let region = Region::new(GLOBAL);
    let mut actions = 0_u64;
    for index in 0..args.rollouts {
        actions += u64::from(
            rollout(
                &context,
                &root,
                args.policy,
                args.seed.wrapping_add(index + 1),
                RolloutLimits::default(),
                &mut scratch,
            )
            .player_actions,
        );
    }
    let stats = region.change();
    Ok(common_report(
        args,
        "allocations",
        Vec::new(),
        json!({
            "rollouts": args.rollouts,
            "player_intents": actions,
            "allocations": stats.allocations,
            "deallocations": stats.deallocations,
            "bytes_allocated": stats.bytes_allocated,
        }),
    ))
}

fn common_report(args: &Args, workload: &str, samples: Vec<f64>, detail: Value) -> Value {
    json!({
        "revision": "plan/rust-rollout-engine",
        "rules_profile": "rust-v1",
        "map": "BASE",
        "policy": format!("{:?}", args.policy).to_lowercase(),
        "seed": args.seed,
        "players": args.players,
        "threads": 1,
        "build": "release-required-for-scoreboard",
        "workload": workload,
        "sample_seconds": samples,
        "detail": detail,
    })
}
