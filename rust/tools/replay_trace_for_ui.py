#!/usr/bin/env python3
"""Force-replay a Rust-decided game trace through the real Python engine.

``export-tournament-replay`` (a Rust binary) plays a full 2-player TOURNAMENT
game entirely inside the Rust engine, both seats using perfect-information
alpha-beta search, and records every (actor, action, chance-outcome) it
picked. This script rebuilds a real ``catanatron.game.Game`` on the identical
fixed TOURNAMENT board and replays that exact trace one action at a time via
``Game.execute(action, action_record=...)`` -- the same "replaying
functionality" hook the engine already uses to rebuild games from a stored
action log. Python is the authority on rules the whole way through; Rust
only supplied the decisions.

After each replayed action it captures ``serialization.web_view(game)`` --
the document shape the web UI (and ui-next's "Import JSON") already
understand -- so the output is a ready-to-import replay of a Rust-vs-Rust
game.

Usage:
    cargo run --release --manifest-path rust/Cargo.toml -p catanatron-bench \
        --bin export-tournament-replay -- --depth 2 --output /tmp/trace.json
    python rust/tools/replay_trace_for_ui.py /tmp/trace.json --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catanatron.game import Game
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer
from catanatron.serialization import web_view

# rust/tests/fixtures/transitions/sample-tournament-4p.jsonl "before.layout"
# lists TOURNAMENT's 19 land tiles in exactly the order that Rust's TileId
# 0..18 indexes (verified: index 0 is Brick@8 and index 9 is the desert,
# matching catanatron_search::initialize_tournament's own unit test). Only
# MOVE_ROBBER needs this -- every other action already addresses nodes and
# edges, which both engines index identically.
TILE_ID_TO_COORDINATE = [
    (-2, 0, 2), (-2, 1, 1), (-2, 2, 0), (-1, -1, 2), (-1, 0, 1), (-1, 1, 0),
    (-1, 2, -1), (0, -2, 2), (0, -1, 1), (0, 0, 0), (0, 1, -1), (0, 2, -2),
    (1, -2, 1), (1, -1, 0), (1, 0, -1), (1, 1, -2), (2, -2, 0), (2, -1, -1),
    (2, 0, -2),
]

RESOURCE = {"WOOD": "WOOD", "BRICK": "BRICK", "SHEEP": "SHEEP", "WHEAT": "WHEAT", "ORE": "ORE"}


def decode_action(entry_actor: int, colors, action: dict) -> Action:
    color = colors[entry_actor]
    kind = action["type"]
    value = action.get("value")
    if kind == "BUILD_ROAD":
        value = tuple(value)
    elif kind == "MOVE_ROBBER":
        tile_id, victim_seat = value
        coordinate = TILE_ID_TO_COORDINATE[tile_id]
        victim = colors[victim_seat] if victim_seat is not None else None
        value = (coordinate, victim)
    elif kind == "PLAY_YEAR_OF_PLENTY":
        first, second = value
        value = (first, second) if second is not None else (first,)
    elif kind == "MARITIME_TRADE":
        value = (value["give"],) * value["rate"] + (None,) * (4 - value["rate"]) + (value["receive"],)
    return Action(color, ActionType[kind], value)


def decode_result(outcome: dict | None):
    """The ``ActionRecord.result`` apply_roll / apply_buy_development_card /
    apply_move_robber force instead of drawing their own randomness. Each of
    those three functions builds its own fully-specified action internally
    (baking the dice / card / stolen resource into ``action.value`` itself);
    the action this script passes to ``Game.execute`` must stay in its
    unresolved menu form (``value=None`` for ROLL and BUY_DEVELOPMENT_CARD)
    so it still matches ``playable_actions``."""
    if outcome is None:
        return None
    kind, value = outcome["type"], outcome["value"]
    if kind == "DICE":
        return tuple(value)
    if kind in ("DEVELOPMENT_CARD", "STOLEN_RESOURCE"):
        return value
    raise ValueError(f"unknown outcome type {kind}")


def replay(trace_path: Path, output_path: Path, turn_limit: int) -> None:
    document = json.loads(trace_path.read_text())
    if document["map"] != "TOURNAMENT" or document["player_count"] != 2:
        raise SystemExit("this script only replays 2-player TOURNAMENT traces")

    # Game() without a seed reshuffles the global `random` module and uses it
    # to decide seating order (state.colors), so replaying the same trace
    # would flip which color is "actor 0" from run to run. Any fixed seed
    # keeps that order (and this script's own results) reproducible.
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players=players, catan_map=build_map("TOURNAMENT"), vps_to_win=10, seed=0)
    colors = game.state.colors  # actual seat order Python assigned, PlayerId-indexed
    states = [web_view(game)]

    for entry in document["trace"]:
        actor = entry["actor"]
        if colors[actor] != game.state.current_color():
            raise AssertionError(
                f"trace actor {colors[actor]} does not match current player "
                f"{game.state.current_color()} at ply {len(states)}"
            )
        action = decode_action(actor, colors, entry["action"])
        result = decode_result(entry["outcome"])
        action_record = ActionRecord(action=action, result=result) if entry["outcome"] else None
        game.execute(action, action_record=action_record)
        states.append(web_view(game))
        if game.state.num_turns > turn_limit:
            raise SystemExit(f"exceeded turn_limit={turn_limit} while replaying")

    winner = game.winning_color()
    expected_winner = colors[document["winner"]] if document["winner"] is not None else None
    if winner != expected_winner:
        raise AssertionError(f"replayed winner {winner} != Rust-reported winner {expected_winner}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"states": states}))
    print(
        f"replayed {len(document['trace'])} actions, {game.state.num_turns} turns, "
        f"winner={winner}; wrote {output_path} ({output_path.stat().st_size} bytes)",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-limit", type=int, default=1000)
    args = parser.parse_args()
    replay(args.trace, args.output, args.turn_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
