"""Run paired screening experiments for configurable placement heuristics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import DEFAULT_WEIGHTS

from benchmark_placement import (
    HybridPlayer,
    _append_jsonl,
    _confidence_interval_bounds,
    _paired_mcnemar_counts,
    run_group,
)
from PlacementAgent import HeuristicPlacementAgent, make_placement_agent
from placement_heuristic import HeuristicConfig, RESOURCE_WEIGHTS


def _weights(**overrides):
    values = RESOURCE_WEIGHTS.copy()
    values.update(overrides)
    return values


def _starting_weights(**overrides):
    values = {
        "WOOD": 1.20,
        "BRICK": 1.20,
        "SHEEP": 1.00,
        "WHEAT": 1.10,
        "ORE": 0.75,
    }
    values.update(overrides)
    return values


def variant_configs() -> dict[str, HeuristicConfig]:
    """Named configs for quick empirical screening."""

    return {
        "current": HeuristicConfig(),
        "start_hand_light": HeuristicConfig(
            starting_resource_weight=0.035,
            starting_resource_weights=_starting_weights(),
        ),
        "start_hand_medium": HeuristicConfig(
            starting_resource_weight=0.070,
            starting_resource_weights=_starting_weights(),
        ),
        "start_hand_heavy": HeuristicConfig(
            starting_resource_weight=0.110,
            starting_resource_weights=_starting_weights(),
        ),
        "denial_light": HeuristicConfig(
            opponent_denial_weight=0.08,
        ),
        "denial_medium": HeuristicConfig(
            opponent_denial_weight=0.16,
        ),
        "start_denial": HeuristicConfig(
            starting_resource_weight=0.060,
            starting_resource_weights=_starting_weights(),
            opponent_denial_weight=0.12,
        ),
        "expansion_balanced": HeuristicConfig(
            resource_weights=_weights(WOOD=1.18, BRICK=1.18, WHEAT=1.15, ORE=1.10, SHEEP=0.92),
            starting_resource_weight=0.060,
            starting_resource_weights=_starting_weights(),
            complementarity_weight=0.13,
            critical_missing_weight=0.05,
            road_new_resource_weight=0.08,
        ),
        "city_heavy": HeuristicConfig(
            resource_weights=_weights(WHEAT=1.45, ORE=1.45, SHEEP=0.85, WOOD=0.95, BRICK=0.95),
            starting_resource_weight=0.045,
            starting_resource_weights=_starting_weights(),
            complementarity_weight=0.08,
            critical_missing_weight=0.08,
            road_new_resource_weight=0.04,
        ),
        "ports_low": HeuristicConfig(
            generic_port_bonus=0.01,
            specific_port_multiplier=0.12,
            specific_port_floor=0.0,
            starting_resource_weight=0.060,
            starting_resource_weights=_starting_weights(),
        ),
        "production_pure": HeuristicConfig(
            variety_weight=0.0,
            complementarity_weight=0.0,
            critical_missing_weight=0.0,
            generic_port_bonus=0.0,
            specific_port_multiplier=0.0,
            specific_port_floor=0.0,
            number_diversity_weight=0.0,
            road_new_resource_weight=0.0,
        ),
    }


SPECIAL_VARIANTS = {
    "value_heuristic",
    "value_heuristic_contender",
    "rollout_value",
    "rollout_value_selfish",
    "rollout_value_first_roll",
    "rollout_value_blend_25",
    "rollout_value_blend_50",
    "rollout_value_blend_75",
    "rollout_value_stable",
    "rollout_value_stable_first_roll",
    "rollout_value_stable_ab_opp",
    "rollout_value_stable_contender",
    "rollout_value_stable_tempo",
    "rollout_value_stable_expansion",
    "rollout_value_stable_denial",
    "rollout_value_stable_public",
    "beam_value",
    "rollout_value_contender",
    "rollout_value_tempo",
    "rollout_value_expansion",
    "rollout_value_denial",
    "rollout_value_public",
}


def value_param_variants() -> dict[str, dict[str, float]]:
    """High-level value-function mutations for rollout placement search."""

    tempo = DEFAULT_WEIGHTS.copy()
    tempo.update(
        {
            "hand_synergy": 5e6,
            "hand_resources": 1e6,
            "buildable_nodes": 5e5,
        }
    )

    expansion = DEFAULT_WEIGHTS.copy()
    expansion.update(
        {
            "reachable_production_0": 1e6,
            "reachable_production_1": 5e6,
            "buildable_nodes": 5e6,
            "longest_road": 1e5,
        }
    )

    denial = DEFAULT_WEIGHTS.copy()
    denial.update(
        {
            "production": 1.0e8,
            "enemy_production": -1.35e8,
            "buildable_nodes": 2e6,
        }
    )

    public = DEFAULT_WEIGHTS.copy()
    public.update(
        {
            "production": 8.0e7,
            "enemy_production": -1.1e8,
            "num_tiles": 5e6,
            "buildable_nodes": 2e6,
            "hand_synergy": 1e6,
        }
    )

    return {
        "rollout_value_tempo": tempo,
        "rollout_value_expansion": expansion,
        "rollout_value_denial": denial,
        "rollout_value_public": public,
        "rollout_value_stable_tempo": tempo,
        "rollout_value_stable_expansion": expansion,
        "rollout_value_stable_denial": denial,
        "rollout_value_stable_public": public,
    }


def blend_weight_variants() -> dict[str, float]:
    """Weights for mixing pre-roll opening value with first-roll expectation."""

    return {
        "rollout_value_blend_25": 0.25,
        "rollout_value_blend_50": 0.50,
        "rollout_value_blend_75": 0.75,
    }


def _summarize_wins(wins: list[bool]):
    n = len(wins)
    count = sum(1 for won in wins if won)
    return {
        "games": n,
        "wins": count,
        "losses": n - count,
        "draws": 0,
        "win_rate": count / n if n else 0.0,
    }


def _load_baseline_cache(path: str | None, games: int, board_seed: int, game_seed_start: int):
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as handle:
        row = json.load(handle)

    if (
        row.get("games") == games
        and row.get("board_seed") == board_seed
        and row.get("game_seed_start") == game_seed_start
    ):
        return [bool(value) for value in row["baseline_wins"]]
    return None


def _save_baseline_cache(
    path: str | None,
    *,
    games: int,
    board_seed: int,
    game_seed_start: int,
    baseline_wins: list[bool],
):
    if not path:
        return

    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "board_seed": board_seed,
                "game_seed_start": game_seed_start,
                "games": games,
                "baseline_wins": baseline_wins,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            sort_keys=True,
        )


def _run_baseline(args):
    cached = _load_baseline_cache(
        args.baseline_cache,
        args.games,
        args.board_seed,
        args.game_seed_start,
    )
    if cached is not None:
        summary = _summarize_wins(cached)
        print(
            "Loaded baseline cache: "
            f"{summary['wins']}W/{summary['losses']}L "
            f"win_rate={summary['win_rate']:.1%}"
        )
        return cached

    print(f"=== Baseline: AlphaBeta placement ({args.games} games) ===")
    _, _, _, _, baseline_wins = run_group(
        "baseline",
        lambda: AlphaBetaPlayer(Color.BLUE),
        args.games,
        board_seed=args.board_seed,
        game_seed_start=args.game_seed_start,
        verbose=args.verbose,
    )
    _save_baseline_cache(
        args.baseline_cache,
        games=args.games,
        board_seed=args.board_seed,
        game_seed_start=args.game_seed_start,
        baseline_wins=baseline_wins,
    )
    return baseline_wins


def _log_variant_result(args, row):
    if args.results_log:
        _append_jsonl(args.results_log, row)


def run_variant(args, name: str, config: HeuristicConfig | None, baseline_wins: list[bool]):
    print(f"\n=== Variant: {name} ({args.games} games) ===")
    config_payload = asdict(config) if config is not None else None
    param_variants = value_param_variants()
    blend_variants = blend_weight_variants()

    if name == "value_heuristic":
        agent = make_placement_agent("value_heuristic")
    elif name == "value_heuristic_contender":
        agent = make_placement_agent(
            "value_heuristic",
            value_fn_builder_name="contender_fn",
        )
    elif name == "rollout_value":
        agent = make_placement_agent("rollout_value")
    elif name == "rollout_value_selfish":
        agent = make_placement_agent("rollout_value_selfish")
    elif name == "rollout_value_first_roll":
        agent = make_placement_agent("rollout_value_first_roll")
    elif name in blend_variants:
        config_payload = {"roll_weight": blend_variants[name]}
        agent = make_placement_agent(
            "rollout_value_blend",
            roll_weight=blend_variants[name],
        )
    elif name == "rollout_value_stable":
        agent = make_placement_agent("rollout_value_stable")
    elif name == "rollout_value_stable_first_roll":
        agent = make_placement_agent("rollout_value_stable_first_roll")
    elif name == "rollout_value_stable_ab_opp":
        agent = make_placement_agent("rollout_value_stable_ab_opp")
    elif name == "rollout_value_stable_contender":
        agent = make_placement_agent(
            "rollout_value_stable",
            value_fn_builder_name="contender_fn",
        )
    elif name == "beam_value":
        agent = make_placement_agent("beam_value")
    elif name == "rollout_value_contender":
        agent = make_placement_agent(
            "rollout_value",
            value_fn_builder_name="contender_fn",
        )
    elif name in param_variants:
        config_payload = param_variants[name]
        strategy = (
            "rollout_value_stable"
            if name.startswith("rollout_value_stable_")
            else "rollout_value"
        )
        agent = make_placement_agent(
            strategy,
            value_fn_builder_name="contender_fn",
            params=param_variants[name],
        )
    else:
        agent = HeuristicPlacementAgent(
            temperature=args.temperature,
            heuristic_config=config,
        )
    wins, losses, draws, elapsed, test_wins = run_group(
        name,
        lambda: HybridPlayer(Color.BLUE, agent),
        args.games,
        board_seed=args.board_seed,
        game_seed_start=args.game_seed_start,
        verbose=args.verbose,
    )

    paired = _paired_mcnemar_counts(baseline_wins, test_wins)
    ci_lo, ci_hi = _confidence_interval_bounds(wins, args.games)
    row = {
        "event": "heuristic_variant",
        "variant": name,
        "board_seed": args.board_seed,
        "game_seed_start": args.game_seed_start,
        "games": args.games,
        "temperature": args.temperature,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / args.games if args.games else 0.0,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "elapsed_seconds": elapsed,
        "config": config_payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row.update(paired)
    _log_variant_result(args, row)

    print(
        f"{name}: {wins}W/{losses}L win_rate={row['win_rate']:.1%} "
        f"b={paired['baseline_only_wins']} c={paired['test_only_wins']} "
        f"p={paired['mcnemar_pvalue']:.4g}"
    )
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Screen placement heuristic variants against a cached paired baseline"
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--board-seed", type=int, default=42)
    parser.add_argument("--game-seed-start", type=int, default=600000)
    parser.add_argument(
        "--baseline-cache",
        type=str,
        default="capstone_agent/models/heuristic_baseline_cache.json",
    )
    parser.add_argument(
        "--results-log",
        type=str,
        default="capstone_agent/models/heuristic_experiment_results.jsonl",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant name to run. Repeat to run several. Defaults to all variants.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configs = variant_configs()
    variant_names = args.variant or list(configs.keys())
    unknown = [
        name
        for name in variant_names
        if name not in configs and name not in SPECIAL_VARIANTS
    ]
    if unknown:
        available = ", ".join(sorted([*configs, *SPECIAL_VARIANTS]))
        raise SystemExit(f"Unknown variant(s): {unknown}. Available: {available}")

    baseline_wins = _run_baseline(args)
    rows = [
        run_variant(args, name, configs.get(name), baseline_wins)
        for name in variant_names
    ]

    print("\n=== Ranked by paired edge (c - b), then win rate ===")
    for row in sorted(
        rows,
        key=lambda r: (
            r["test_only_wins"] - r["baseline_only_wins"],
            r["win_rate"],
        ),
        reverse=True,
    ):
        edge = row["test_only_wins"] - row["baseline_only_wins"]
        print(
            f"{row['variant']:22s} edge={edge:+4d} "
            f"win_rate={row['win_rate']:.1%} "
            f"b={row['baseline_only_wins']} c={row['test_only_wins']} "
            f"p={row['mcnemar_pvalue']:.4g}"
        )


if __name__ == "__main__":
    main()
