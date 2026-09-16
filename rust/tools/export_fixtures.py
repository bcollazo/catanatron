#!/usr/bin/env python3
"""Export deterministic, canonical Python transition fixtures for Rust."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from catanatron.game import Game
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, Action
from catanatron.models.enums import ActionPrompt, ActionType, CITY, SETTLEMENT
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, build_map
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.state_functions import build_settlement, player_deck_replenish

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
REVISION = "d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816"
PROFILE = "rust-v1"


def normalize(value: Any) -> Any:
    if isinstance(value, Color):
        return value.value
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def action_value(action: Action, *, intent: bool = False) -> dict[str, Any]:
    result = {"color": action.color.value, "type": action.action_type.value, "value": normalize(action.value)}
    if intent and result["type"] in {"ROLL", "BUY_DEVELOPMENT_CARD"}:
        result["value"] = None
    return result


def menu(actions: list[Action]) -> list[dict[str, Any]]:
    return sorted((action_value(action) for action in actions), key=lambda item: json.dumps(item, sort_keys=True))


def snapshot(game: Game, map_name: str) -> dict[str, Any]:
    state = game.state
    board = state.board
    colors = [color.value for color in state.colors]
    players = []
    for index, color in enumerate(state.colors):
        prefix = f"P{index}_"
        players.append(
            {
                "color": color.value,
                "hand": [state.player_state[prefix + f"{resource}_IN_HAND"] for resource in RESOURCES],
                "dev": [state.player_state[prefix + f"{card}_IN_HAND"] for card in DEVELOPMENT_CARDS],
                "eligible_dev": [state.player_state[prefix + f"{card}_OWNED_AT_START"] for card in DEVELOPMENT_CARDS[:-1]],
                "played_dev": state.player_state[prefix + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
                "pieces": [state.player_state[prefix + "ROADS_AVAILABLE"], state.player_state[prefix + "SETTLEMENTS_AVAILABLE"], state.player_state[prefix + "CITIES_AVAILABLE"]],
                "played_knights": state.player_state[prefix + "PLAYED_KNIGHT"],
                # The player-state cache is not refreshed for free setup roads;
                # Board owns the graph-derived semantic value used by the rules.
                "longest_road_length": board.road_lengths[color],
                "has_longest_road": state.player_state[prefix + "HAS_ROAD"],
                "has_largest_army": state.player_state[prefix + "HAS_ARMY"],
                "has_rolled": state.player_state[prefix + "HAS_ROLLED"],
            }
        )
    buildings = [None] * 54
    for node, (color, kind) in board.buildings.items():
        buildings[node] = [color.value, kind]
    roads = [
        [list(edge), color.value]
        for edge, color in sorted(board.roads.items())
        if edge[0] < edge[1]
    ]
    ports = [
        {
            "resource": port.resource,
            "nodes": sorted(
                port.nodes[node_ref]
                for node_ref in PORT_DIRECTION_TO_NODEREFS[port.direction]
            ),
        }
        for _, port in sorted(board.map.ports_by_id.items())
    ]
    layout = [
        {"coordinate": list(coord), "resource": tile.resource, "number": tile.number}
        for coord, tile in sorted(board.map.land_tiles.items())
    ]
    return {
        "map": map_name,
        "layout": layout,
        "colors": colors,
        "bank": list(state.resource_freqdeck),
        "development_deck": list(state.development_listdeck),
        "players": players,
        "buildings": buildings,
        "roads": roads,
        "ports": ports,
        "robber": list(board.robber_coordinate),
        "actor": state.current_color().value,
        "turn_owner": state.colors[state.current_turn_index].value,
        "prompt": state.current_prompt.value,
        "phase": phase_value(game),
        "initial_build": state.is_initial_build_phase,
        "discard_counts": list(state.discard_counts),
        "road_building": state.free_roads_available,
        "trade": normalize(state.current_trade),
        "acceptees": list(state.acceptees),
        "responded": trade_responded(state),
        "turns": state.num_turns,
        "friendly_robber": state.friendly_robber,
    }


def trade_responded(state) -> list[bool]:
    responded = [False] * len(state.colors)
    if not state.is_resolving_trade:
        return responded
    proposer = state.current_trade[10]
    if state.current_prompt == ActionPrompt.DECIDE_ACCEPTEES:
        return [color != proposer for color in state.colors]
    if state.current_prompt == ActionPrompt.DECIDE_TRADE:
        for index in range(state.current_player_index):
            responded[index] = state.colors[index] != proposer
    return responded


def phase_value(game: Game) -> dict[str, Any]:
    """Export the semantic phase; Python's deprecated prompt is not sufficient."""
    state = game.state
    actor = state.current_color().value
    winner = game.winning_color()
    if winner is not None:
        return {"kind": "TERMINAL", "winner": winner.value}
    turn_prefix = f"P{state.current_turn_index}_"
    resume_post_roll = state.player_state[turn_prefix + "HAS_ROLLED"]
    if state.is_initial_build_phase:
        kind = (
            "SETUP_SETTLEMENT"
            if state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
            else "SETUP_ROAD"
        )
        building_count = len(state.board.buildings)
        phase = {
            "kind": kind,
            "actor": actor,
            "reverse": (
                building_count >= len(state.colors)
                if kind == "SETUP_SETTLEMENT"
                else building_count > len(state.colors)
            ),
        }
        if kind == "SETUP_ROAD":
            phase["settlement"] = state.action_records[-1].action.value
        return phase
    if state.current_prompt == ActionPrompt.DISCARD:
        return {
            "kind": "DISCARD",
            "actor": actor,
            "remaining": state.discard_counts[state.current_player_index],
        }
    if state.current_prompt == ActionPrompt.MOVE_ROBBER:
        return {"kind": "ROBBER", "actor": actor, "resume_post_roll": resume_post_roll}
    if state.is_road_building and state.free_roads_available:
        return {
            "kind": "FREE_ROAD",
            "actor": actor,
            "remaining": state.free_roads_available,
            "resume_post_roll": resume_post_roll,
        }
    if state.current_prompt == ActionPrompt.DECIDE_TRADE:
        return {"kind": "TRADE_RESPONSE", "actor": actor}
    if state.current_prompt == ActionPrompt.DECIDE_ACCEPTEES:
        return {"kind": "CHOOSE_ACCEPTER", "actor": actor}
    return {"kind": "POST_ROLL" if resume_post_roll else "PRE_ROLL", "actor": actor}


def trace(case_id: str, players: int, map_name: str, seed: int, limit: int) -> list[dict[str, Any]]:
    random.seed(seed)
    colors = list(Color)[:players]
    # `build_map("BASE")` consumes Python's module-level RNG. Seed it before
    # construction as well as passing the game seed, otherwise `--check`
    # would compare different board assignments on each invocation.
    catan_map = build_map(map_name)
    game = Game([RandomPlayer(color) for color in colors], seed=seed, catan_map=catan_map)
    rows = []
    for step in range(limit):
        if game.winning_color() is not None:
            break
        before = snapshot(game, map_name)
        legal_before = menu(game.playable_actions)
        # Python's move helpers use sets for some menus, so their list order is
        # process-hash-dependent. Select from a canonical ordering to make
        # fixtures reproducible across interpreter processes.
        ordered = sorted(game.playable_actions, key=lambda item: json.dumps(action_value(item), sort_keys=True))
        action = ordered[step % len(ordered)]
        record = game.execute(action)
        outcome = normalize(record.result) if record.result is not None else None
        rows.append(
            {
                "fixture_version": 2,
                "case_id": f"{case_id}-{step:04d}",
                "source_revision": REVISION,
                "rules_profile": PROFILE,
                "before": before,
                "actor": action.color.value,
                "action": action_value(action, intent=True),
                "outcome": outcome,
                "after": snapshot(game, map_name),
                "legal_before": legal_before,
                "legal_after": menu(game.playable_actions),
                "status_after": "won" if game.winning_color() else "decision",
            }
        )
    return rows


def record_transition(game: Game, map_name: str, case_id: str, action: Action) -> dict[str, Any]:
    before = snapshot(game, map_name)
    legal_before = menu(game.playable_actions)
    record = game.execute(action)
    return {
        "fixture_version": 2,
        "case_id": case_id,
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "before": before,
        "actor": action.color.value,
        "action": action_value(action, intent=True),
        "outcome": normalize(record.result) if record.result is not None else None,
        "after": snapshot(game, map_name),
        "legal_before": legal_before,
        "legal_after": menu(game.playable_actions),
        "status_after": "won" if game.winning_color() else "decision",
    }


def prepared_post_roll(players: int, seed: int) -> Game:
    """Make a small legal post-roll state without relying on trace ordering."""
    random.seed(seed)
    colors = list(Color)[:players]
    game = Game([SimplePlayer(color) for color in colors], seed=seed)
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.current_player_index = 0
    state.current_turn_index = 0
    state.player_state["P0_HAS_ROLLED"] = True
    game.playable_actions = []
    return game


def crafted_city() -> list[dict[str, Any]]:
    game = prepared_post_roll(2, 101)
    state, color = game.state, game.state.colors[0]
    state.board.build_settlement(color, 0, initial_build_phase=True)
    build_settlement(state, color, 0, True)
    player_deck_replenish(state, color, "WHEAT", 2)
    player_deck_replenish(state, color, "ORE", 3)
    from catanatron.models.actions import generate_playable_actions

    game.playable_actions = generate_playable_actions(state)
    action = Action(color, ActionType.BUILD_CITY, 0)
    return [record_transition(game, "BASE", "crafted-city", action)]


def crafted_trades() -> list[dict[str, Any]]:
    from catanatron.models.actions import generate_playable_actions

    rows: list[dict[str, Any]] = []
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    # Accept then confirm: every trade message is represented as its own intent.
    game = prepared_post_roll(3, 102)
    state, proposer, accepter = game.state, game.state.colors[0], game.state.colors[1]
    player_deck_replenish(state, proposer, "WOOD")
    player_deck_replenish(state, accepter, "BRICK")
    game.playable_actions = generate_playable_actions(state)
    rows.append(record_transition(game, "BASE", "crafted-trade-offer", Action(proposer, ActionType.OFFER_TRADE, offer)))
    accept = next(action for action in game.playable_actions if action.action_type == ActionType.ACCEPT_TRADE)
    rows.append(record_transition(game, "BASE", "crafted-trade-accept", accept))
    responder = game.state.current_color()
    reject = next(action for action in game.playable_actions if action.action_type == ActionType.REJECT_TRADE)
    rows.append(record_transition(game, "BASE", "crafted-trade-reject-after-accept", reject))
    confirm = next(action for action in game.playable_actions if action.action_type == ActionType.CONFIRM_TRADE)
    rows.append(record_transition(game, "BASE", "crafted-trade-confirm", confirm))

    # A separate offer/reject sequence exercises the no-accepter return path.
    game = prepared_post_roll(2, 103)
    state, proposer, responder = game.state, game.state.colors[0], game.state.colors[1]
    player_deck_replenish(state, proposer, "WOOD")
    game.playable_actions = generate_playable_actions(state)
    rows.append(record_transition(game, "BASE", "crafted-trade-offer-reject", Action(proposer, ActionType.OFFER_TRADE, offer)))
    reject = next(action for action in game.playable_actions if action.action_type == ActionType.REJECT_TRADE)
    rows.append(record_transition(game, "BASE", "crafted-trade-reject", reject))

    # A single accepted offer can be cancelled by its proposer.
    game = prepared_post_roll(2, 104)
    state, proposer, accepter = game.state, game.state.colors[0], game.state.colors[1]
    player_deck_replenish(state, proposer, "WOOD")
    player_deck_replenish(state, accepter, "BRICK")
    game.playable_actions = generate_playable_actions(state)
    rows.append(record_transition(game, "BASE", "crafted-trade-offer-cancel", Action(proposer, ActionType.OFFER_TRADE, offer)))
    accept = next(action for action in game.playable_actions if action.action_type == ActionType.ACCEPT_TRADE)
    rows.append(record_transition(game, "BASE", "crafted-trade-accept-cancel", accept))
    rows.append(record_transition(game, "BASE", "crafted-trade-cancel", Action(proposer, ActionType.CANCEL_TRADE, None)))
    return rows


def named_divergences() -> dict[Path, str]:
    """Capture deliberately corrected `rust-v1` behavior with Python evidence."""
    # Python's response-advance implementation revisits a proposer seated at
    # index 1 after seat 0 rejects. `rust-v1` instead visits each *other* seat
    # exactly once in ascending seat order.
    game = prepared_post_roll(3, 105)
    state = game.state
    proposer = state.colors[1]
    state.current_player_index = state.current_turn_index = 1
    state.player_state["P1_HAS_ROLLED"] = True
    player_deck_replenish(state, proposer, "WOOD")
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    from catanatron.models.actions import generate_playable_actions

    game.playable_actions = generate_playable_actions(state)
    game.execute(Action(proposer, ActionType.OFFER_TRADE, offer))
    first_responder = state.current_color()
    first_reject = next(action for action in game.playable_actions if action.action_type == ActionType.REJECT_TRADE)
    game.execute(first_reject)
    if state.current_color() != proposer:
        raise AssertionError("expected pinned Python trade advance to revisit proposer")
    trade = {
        "divergence_id": "D001-domestic-trade-proposer-revisited",
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "input": {"seat_order": [color.value for color in state.colors], "proposer_index": 1, "first_responder": first_responder.value},
        "python_observed": {"next_actor": proposer.value, "prompt": state.current_prompt.value},
        "rust_expected": {"next_actor": state.colors[2].value, "prompt": "DECIDE_TRADE"},
        "rationale": "Each other seat responds once; the proposer is never asked to answer its own offer.",
    }
    longest_road = {
        "divergence_id": "D002-longest-road-entering-opponent-building",
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "input": {
            "actor": "ORANGE",
            "owned_roads_before": [[35, 36], [36, 37]],
            "built_road": [37, 38],
            "opponent_building_node": 38,
        },
        "python_observed": {"longest_road_length": 2},
        "rust_expected": {"longest_road_length": 3},
        "rationale": "An edge-simple trail may enter, but not continue through, an opponent building. The pinned Python connected-component cache drops the incoming edge; rust-v1 follows the explicit E02 correction.",
    }
    longest_road_tie = {
        "divergence_id": "D004-longest-road-incumbent-tie-retention",
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "input": {
            "lengths_before": [5, 4, 4, 7],
            "holder_before": "ORANGE",
            "lengths_after": [5, 4, 4, 5],
            "trigger": "RED settlement splits ORANGE road",
        },
        "python_observed": {"holder": "RED"},
        "rust_expected": {"holder": "ORANGE"},
        "rationale": "rust-v1 follows E06: an incumbent tied for the maximum at or above five retains Longest Road. The pinned Python cache transfers to the first maximum after a road split.",
    }
    longest_road_branch = {
        "divergence_id": "D003-longest-road-branch-undercount",
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "input": {
            "actor": "RED",
            "owned_roads_after": [[0, 20], [19, 20], [19, 21], [20, 22], [22, 23], [23, 52]],
            "edge_simple_trail": [[19, 21], [19, 20], [20, 22], [22, 23], [23, 52]],
        },
        "python_observed": {"longest_road_length": 4, "holder": None},
        "rust_expected": {"longest_road_length": 5, "holder": "RED"},
        "rationale": "The pinned Python traversal undercounts this branching graph. rust-v1 uses the E02 edge-simple-trail definition and retains all five edges in the valid trail.",
    }
    longest_road_threshold = {
        "divergence_id": "D005-longest-road-below-threshold-award",
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "input": {
            "lengths_before": [2, 4, 3, 4],
            "lengths_after": [2, 2, 3, 4],
            "trigger": "BLUE settlement splits RED road",
        },
        "python_observed": {"holder": "BLUE", "holder_length": 4},
        "rust_expected": {"holder": None},
        "rationale": "rust-v1 enforces the E06 minimum length of five. The pinned Python maintenance path can award the new maximum after a split without rechecking that threshold.",
    }

    return {
        OUT / "divergences" / "D001-domestic-trade-proposer-revisited.json": json.dumps(trade, indent=2, sort_keys=True) + "\n",
        OUT / "divergences" / "D002-longest-road-entering-opponent-building.json": json.dumps(longest_road, indent=2, sort_keys=True) + "\n",
        OUT / "divergences" / "D004-longest-road-incumbent-tie-retention.json": json.dumps(longest_road_tie, indent=2, sort_keys=True) + "\n",
        OUT / "divergences" / "D003-longest-road-branch-undercount.json": json.dumps(longest_road_branch, indent=2, sort_keys=True) + "\n",
        OUT / "divergences" / "D005-longest-road-below-threshold-award.json": json.dumps(longest_road_threshold, indent=2, sort_keys=True) + "\n",
    }


def write_or_check(path: Path, data: str, check: bool) -> bool:
    if check:
        return path.exists() and path.read_text(encoding="utf-8") == data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int, default=96)
    args = parser.parse_args()
    files: dict[Path, str] = {}
    coverage: Counter[str] = Counter()
    for players, map_name, seed in (
        (2, "BASE", 11),
        (3, "BASE", 17),
        (4, "BASE", 23),
        (4, "TOURNAMENT", 29),
        (2, "MINI", 31),
        (3, "MINI", 37),
        (4, "MINI", 41),
    ):
        rows = trace(f"sample-{map_name.lower()}-{players}p", players, map_name, seed, args.limit)
        for row in rows:
            coverage[row["action"]["type"]] += 1
        files[OUT / "transitions" / f"sample-{map_name.lower()}-{players}p.jsonl"] = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    crafted = crafted_city() + crafted_trades()
    for row in crafted:
        coverage[row["action"]["type"]] += 1
    files[OUT / "transitions" / "crafted-builds-and-trades.jsonl"] = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in crafted)
    files.update(named_divergences())
    chance = {
        "fixture_version": 2,
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "dice": [{"pair": [first, second], "sum": first + second, "weight": 1} for first in range(1, 7) for second in range(1, 7)],
        "dice_sum_weights": {str(total): 6 - abs(7 - total) for total in range(2, 13)},
        "theft": {"hand": [2, 0, 1, 0, 3], "outcomes": [{"resource": resource, "weight": count} for resource, count in zip(RESOURCES, [2, 0, 1, 0, 3]) if count]},
        "development_draw": {"counts": {"KNIGHT": 14, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2, "ROAD_BUILDING": 2, "VICTORY_POINT": 5}, "denominator": 25},
    }
    files[OUT / "transitions" / "chance-outcomes.json"] = json.dumps(chance, indent=2, sort_keys=True) + "\n"
    manifest = {
        "fixture_version": 2,
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "resource_order": RESOURCES,
        "development_order": DEVELOPMENT_CARDS,
        "coverage": dict(sorted(coverage.items())),
        "canonical_state": "typed phase, owned roads, ports, awards, and resume state",
        "files": {
            path.relative_to(OUT).as_posix(): hashlib.sha256(contents.encode()).hexdigest()
            for path, contents in sorted(files.items())
        },
    }
    files[OUT / "manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    failed = [str(path) for path, contents in files.items() if not write_or_check(path, contents, args.check)]
    if failed:
        raise SystemExit("fixture artifacts differ: " + ", ".join(failed))
    print("fixture artifacts " + ("match" if args.check else "written") + f"; sampled action types: {', '.join(sorted(coverage))}")


if __name__ == "__main__":
    main()
