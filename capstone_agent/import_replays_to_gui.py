"""Import saved game JSON replays into the GUI database.

This loads replay JSON files (saved with GameEncoder) and writes every
intermediate state to the DB so they can be stepped through at:
http://localhost:3000/replays/<game_id>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

# Allow running from repo root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
CATANATRON_SRC = REPO_ROOT / "catanatron"
if str(CATANATRON_SRC) not in sys.path:
    sys.path.insert(0, str(CATANATRON_SRC))

from catanatron.game import Game
from catanatron.json import action_from_json
from catanatron.models.enums import ActionRecord
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    CatanMap,
    LandTile,
    Port,
    Water,
    initialize_tiles,
)
from catanatron.models.player import Color, SimplePlayer
from catanatron.web.models import GameState, database_session, upsert_game_state


@contextlib.contextmanager
def _preserve_player_order():
    """Disable seat randomization while constructing a replay Game."""
    original_sample = random.sample
    random.sample = lambda seq, n: list(seq)[:n]
    try:
        yield
    finally:
        random.sample = original_sample


def _coord_key(coordinate: Iterable[int]) -> tuple[int, int, int]:
    x, y, z = coordinate
    return int(x), int(y), int(z)


def _detect_template(tile_lookup: dict[tuple[int, int, int], dict]):
    coords = set(tile_lookup.keys())
    for name, template in (("BASE", BASE_MAP_TEMPLATE), ("MINI", MINI_MAP_TEMPLATE)):
        if coords == set(template.topology.keys()):
            return name, template
    raise ValueError("Could not match replay tiles to BASE or MINI map template")


def _extract_shuffle_arrays(template, tile_lookup: dict[tuple[int, int, int], dict]):
    numbers: list[int] = []
    port_resources: list[str | None] = []
    tile_resources: list[str | None] = []

    for coordinate, topo in template.topology.items():
        tile = tile_lookup.get(coordinate)
        if tile is None:
            raise ValueError(f"Missing tile at coordinate {coordinate}")
        tile_type = tile.get("type")

        if topo is LandTile:
            if tile_type == "DESERT":
                tile_resources.append(None)
            elif tile_type == "RESOURCE_TILE":
                resource = tile.get("resource")
                number = tile.get("number")
                if resource is None or number is None:
                    raise ValueError(
                        f"Invalid resource tile data at {coordinate}: {tile}"
                    )
                tile_resources.append(resource)
                numbers.append(int(number))
            else:
                raise ValueError(
                    f"Expected land tile at {coordinate}, got type={tile_type}"
                )
        elif isinstance(topo, tuple) and topo[0] is Port:
            if tile_type != "PORT":
                raise ValueError(
                    f"Expected port tile at {coordinate}, got type={tile_type}"
                )
            port_resources.append(tile.get("resource"))
        elif topo is Water:
            if tile_type != "WATER":
                raise ValueError(
                    f"Expected water tile at {coordinate}, got type={tile_type}"
                )
        else:
            raise ValueError(f"Unsupported topology entry at {coordinate}: {topo}")

    # initialize_tiles() consumes these arrays via .pop(), so replay arrays must
    # be in reverse insertion order to reproduce exact coordinate assignments.
    return list(reversed(numbers)), list(reversed(port_resources)), list(
        reversed(tile_resources)
    )


def _build_map_from_payload(payload: dict) -> CatanMap:
    tile_lookup = {
        _coord_key(tile_info["coordinate"]): tile_info["tile"]
        for tile_info in payload["tiles"]
    }
    _, template = _detect_template(tile_lookup)
    numbers, port_resources, tile_resources = _extract_shuffle_arrays(
        template, tile_lookup
    )
    tiles = initialize_tiles(template, numbers, port_resources, tile_resources)
    return CatanMap.from_tiles(tiles)


def _build_players(colors: list[str]):
    return [SimplePlayer(Color[color]) for color in colors]


def import_replay_json(
    json_path: Path,
    replace_existing: bool = True,
    id_prefix: str | None = None,
    skip_invalid_actions: bool = False,
    replay_source_folder: str | None = None,
    imported_at_utc: str | None = None,
):
    payload = json.loads(json_path.read_text())
    players = _build_players(payload["colors"])
    catan_map = _build_map_from_payload(payload)
    with _preserve_player_order():
        game = Game(players=players, catan_map=catan_map)

    game_id = json_path.stem if id_prefix is None else f"{id_prefix}-{json_path.stem}"
    game.id = game_id

    with database_session() as session:
        if replace_existing:
            session.query(GameState).filter_by(uuid=game_id).delete()
            session.commit()

        # Initial state (index 0)
        upsert_game_state(
            game,
            session,
            replay_source_folder=replay_source_folder,
            imported_at_utc=imported_at_utc,
        )

        # One DB row per action so GUI can step through every state.
        imported_actions = 0
        skipped_rows: list[dict] = []
        for idx, (action_json, result) in enumerate(payload["action_records"]):
            action = action_from_json(action_json)
            action_record = ActionRecord(action=action, result=result)
            try:
                game.execute(action, validate_action=False, action_record=action_record)
                imported_actions += 1
                upsert_game_state(
                    game,
                    session,
                    replay_source_folder=replay_source_folder,
                    imported_at_utc=imported_at_utc,
                )
            except Exception as exc:
                if not skip_invalid_actions:
                    raise RuntimeError(
                        f"action_index={idx} action={action_json} error={exc}"
                    ) from exc
                skipped_rows.append(
                    {"index": idx, "action": action_json, "error": str(exc)}
                )

    action_count = len(payload["action_records"])
    winning_color = payload.get("winning_color")
    if winning_color is None:
        result_label = "IN_PROGRESS"
    else:
        result_label = f"WINNER_{winning_color}"

    return {
        "file": json_path.name,
        "game_id": game_id,
        "actions": imported_actions,
        "states": imported_actions + 1,
        "total_actions": action_count,
        "skipped_actions": len(skipped_rows),
        "winner": winning_color,
        "result": result_label,
        "state_index": payload.get("state_index"),
        "url": f"http://localhost:3000/replays/{game_id}",
        "replay_source_folder": replay_source_folder,
        "imported_at_utc": imported_at_utc,
        "skipped_action_details": skipped_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import replay JSON files into GUI database for step-through replay."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing replay JSON files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.json",
        help="Glob pattern inside --input-dir (default: *.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of files to import (0 = no limit).",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not delete existing DB rows for the same game UUID before import.",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default=None,
        help="Optional prefix for imported game UUIDs.",
    )
    parser.add_argument(
        "--skip-invalid-actions",
        action="store_true",
        help=(
            "Continue importing when an action fails to replay. "
            "Failed actions are skipped and summarized."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    paths = sorted(input_dir.glob(args.glob))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No files matched {args.glob!r} in {input_dir}")

    imported = 0
    imported_rows: list[dict] = []
    failed_rows: list[tuple[str, str]] = []
    for path in paths:
        try:
            imported_at_utc = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
            row = import_replay_json(
                path,
                replace_existing=not args.no_replace,
                id_prefix=args.id_prefix,
                skip_invalid_actions=args.skip_invalid_actions,
                replay_source_folder=str(path.parent),
                imported_at_utc=imported_at_utc,
            )
            imported += 1
            imported_rows.append(row)
            print(
                f"[{imported}/{len(paths)}] imported {row['file']} -> "
                f"{row['url']} ({row['states']} states, result={row['result']}, "
                f"skipped={row['skipped_actions']})"
            )
        except Exception as exc:
            failed_rows.append((path.name, str(exc)))
            print(f"[ERROR] {path.name}: {exc}")

    print(f"Done. Imported {imported}/{len(paths)} replay file(s).")

    if imported_rows:
        print("\nImported Replay Links:")
        for row in imported_rows:
            print(
                f"- {row['file']} | result={row['result']} | "
                f"winner={row['winner']} | actions={row['actions']}/{row['total_actions']} | "
                f"states={row['states']} | state_index={row['state_index']} | "
                f"folder={row['replay_source_folder']} | imported_at={row['imported_at_utc']} | "
                f"{row['url']}"
            )
            if row["skipped_actions"] > 0:
                print(f"  skipped_actions={row['skipped_actions']}")
                for detail in row["skipped_action_details"][:3]:
                    print(
                        "  - "
                        f"idx={detail['index']} action={detail['action']} "
                        f"error={detail['error']}"
                    )
                extra = row["skipped_actions"] - 3
                if extra > 0:
                    print(f"  - ... and {extra} more skipped actions")

    if failed_rows:
        print("\nFailed Imports:")
        for file_name, error in failed_rows:
            print(f"- {file_name} | error={error}")


if __name__ == "__main__":
    main()
