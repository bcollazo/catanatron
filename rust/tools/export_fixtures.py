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
from catanatron.models.map import build_map
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
        "robber": list(board.robber_coordinate),
        "actor": state.current_color().value,
        "turn_owner": state.colors[state.current_turn_index].value,
        "prompt": state.current_prompt.value,
        "initial_build": state.is_initial_build_phase,
        "discard_counts": list(state.discard_counts),
        "road_building": state.free_roads_available,
        "trade": normalize(state.current_trade),
        "acceptees": list(state.acceptees),
        "turns": state.num_turns,
    }


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
                "fixture_version": 1,
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
        "fixture_version": 1,
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
    for players, map_name, seed in ((2, "BASE", 11), (3, "BASE", 17), (4, "BASE", 23), (4, "TOURNAMENT", 29)):
        rows = trace(f"sample-{map_name.lower()}-{players}p", players, map_name, seed, args.limit)
        for row in rows:
            coverage[row["action"]["type"]] += 1
        files[OUT / "transitions" / f"sample-{map_name.lower()}-{players}p.jsonl"] = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    crafted = crafted_city() + crafted_trades()
    for row in crafted:
        coverage[row["action"]["type"]] += 1
    files[OUT / "transitions" / "crafted-builds-and-trades.jsonl"] = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in crafted)
    manifest = {
        "fixture_version": 1,
        "source_revision": REVISION,
        "rules_profile": PROFILE,
        "resource_order": RESOURCES,
        "development_order": DEVELOPMENT_CARDS,
        "coverage": dict(sorted(coverage.items())),
        "known_incomplete_coverage": "sample traces only; crafted action/phase/divergence fixtures are still required for E02",
        "files": {str(path.relative_to(OUT)): hashlib.sha256(contents.encode()).hexdigest() for path, contents in sorted(files.items())},
    }
    files[OUT / "manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    failed = [str(path) for path, contents in files.items() if not write_or_check(path, contents, args.check)]
    if failed:
        raise SystemExit("fixture artifacts differ: " + ", ".join(failed))
    print("fixture artifacts " + ("match" if args.check else "written") + f"; sampled action types: {', '.join(sorted(coverage))}")


if __name__ == "__main__":
    main()
