use std::{env, process::ExitCode, time::Duration};

use catanatron_core::{actual_victory_points, GameContext, PlayerId, Position};
use catanatron_search::{
    initialize_base, iterative_alpha_beta_with_evaluator, AlphaBetaMode, NumberPlacement,
};
use serde_json::json;

fn main() -> ExitCode {
    let budget_ms = env::args()
        .nth(1)
        .unwrap_or_else(|| "1000".to_owned())
        .parse::<u64>()
        .expect("usage: alpha-depth [BUDGET_MS] [MAX_DEPTH]");
    let max_depth = env::args()
        .nth(2)
        .unwrap_or_else(|| "32".to_owned())
        .parse::<u8>()
        .expect("MAX_DEPTH must fit in a byte");
    let (context, root) =
        initialize_base(2, NumberPlacement::OfficialSpiral, 91, 0).expect("valid game");
    let result = iterative_alpha_beta_with_evaluator(
        &context,
        &root,
        max_depth,
        Duration::from_millis(budget_ms),
        AlphaBetaMode::Full,
        |_context: &GameContext, position: &Position, player: PlayerId| {
            f64::from(actual_victory_points(position, player))
        },
    );
    println!(
        "{}",
        json!({
            "engine": "rust",
            "position": "two-player BASE opening, seed 91, official spiral",
            "budget_ms": budget_ms,
            "max_depth": max_depth,
            "completed_depth": result.stats.completed_depth,
            "attempted_depth": result.stats.attempted_depth,
            "nodes": result.stats.nodes,
            "chance_children": result.stats.chance_children,
            "elapsed_ms": result.stats.elapsed.as_secs_f64() * 1000.0,
            "heuristic": "actual victory points only",
        })
    );
    ExitCode::SUCCESS
}
