"""Full-fidelity JSON serialization of a game.

``state_to_json`` produces a document that completely defines a game: feeding
it to ``state_from_json`` reconstructs a ``Game`` that plays on identically.
This is what lets games be persisted without pickle.

Two documents, deliberately different:

- ``state_to_json(game)`` is authoritative and server-side only. It includes
  hidden information, most importantly the order of the development card deck.
- ``client_view(doc)`` is the redacted projection handed to a browser or to an
  out-of-process bot. It replaces the ordered deck with its composition
  (how many of each card remain), which is what a player legitimately knows
  and all that the builtin bots use; see
  ``catanatron.players.tree_search_utils.execute_spectrum``.
  Given a ``perspective`` it also hides what only the other seats know, so a
  bot on the wire sees exactly what a person across the table would.

None of this is on the path of an in-process bot, which reads ``game.state``
directly; a self-play run never serializes anything.
"""

from collections import Counter, defaultdict

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import STATIC_GRAPH, Board
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Action,
    ActionPrompt,
    ActionRecord,
    ActionType,
)
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    CatanMap,
    LandTile,
    Port,
    Water,
    get_nodes_and_edges,
)
from catanatron.models.player import Color
from catanatron.state import PLAYER_INITIAL_STATE, State
from catanatron.state_functions import get_longest_road_length

#: Bumped whenever the document shape changes incompatibly.
SCHEMA_VERSION = 1

MAP_TEMPLATES = {"BASE": BASE_MAP_TEMPLATE, "MINI": MINI_MAP_TEMPLATE}


# ===== map =====
def detect_template(catan_map: CatanMap) -> str:
    """Infer which template a map was built from, by its topology.

    A TOURNAMENT map is a BASE topology with a fixed tile assignment, so it
    round-trips correctly as BASE plus the assignment stored below.
    """
    coordinates = set(catan_map.tiles)
    for name, template in MAP_TEMPLATES.items():
        if coordinates == set(template.topology):
            return name
    raise ValueError("map does not match any known template topology")


def map_to_json(catan_map: CatanMap, template_name=None):
    """Store the tile ASSIGNMENT only; node/edge geometry is derived from the
    template topology, which is deterministic."""
    template_name = template_name or detect_template(catan_map)
    tiles = []
    for coord, tile in catan_map.tiles.items():
        if isinstance(tile, LandTile):
            tiles.append(
                {
                    "coordinate": list(coord),
                    "type": "LAND",
                    "id": tile.id,
                    "resource": tile.resource,
                    "number": tile.number,
                }
            )
        elif isinstance(tile, Port):
            # direction is static per template topology -> not stored
            tiles.append(
                {
                    "coordinate": list(coord),
                    "type": "PORT",
                    "id": tile.id,
                    "resource": tile.resource,
                }
            )
        else:
            tiles.append({"coordinate": list(coord), "type": "WATER"})
    return {"template": template_name, "tiles": tiles}


def map_from_json(doc) -> CatanMap:
    template = MAP_TEMPLATES[doc["template"]]
    by_coord = {tuple(t["coordinate"]): t for t in doc["tiles"]}
    all_tiles, node_autoinc = {}, 0
    for coordinate, tile_type in template.topology.items():
        nodes, edges, node_autoinc = get_nodes_and_edges(
            all_tiles, coordinate, node_autoinc
        )
        stored = by_coord[coordinate]
        if stored["type"] == "PORT":
            _, direction = tile_type  # from the template topology
            all_tiles[coordinate] = Port(
                stored["id"], stored["resource"], direction, nodes, edges
            )
        elif stored["type"] == "LAND":
            all_tiles[coordinate] = LandTile(
                stored["id"], stored["resource"], stored["number"], nodes, edges
            )
        else:
            all_tiles[coordinate] = Water(nodes, edges)
    return CatanMap.from_tiles(all_tiles)


# ===== board =====
def board_to_json(board: Board):
    return {
        "buildings": [[nid, c.value, bt] for nid, (c, bt) in board.buildings.items()],
        "roads": [[list(e), c.value] for e, c in board.roads.items()],
        "robber_coordinate": list(board.robber_coordinate),
        "connected_components": {
            c.value: [sorted(s) for s in comps]
            for c, comps in board.connected_components.items()
        },
        "board_buildable_ids": sorted(board.board_buildable_ids),
        "road_lengths": {c.value: n for c, n in board.road_lengths.items()},
        "road_color": board.road_color.value if board.road_color else None,
        "road_length": board.road_length,
    }


def board_from_json(doc, catan_map: CatanMap) -> Board:
    board = Board(catan_map, initialize=False)
    board.map = catan_map
    board.buildings = {nid: (Color[c], bt) for nid, c, bt in doc["buildings"]}
    board.roads = {tuple(e): Color[c] for e, c in doc["roads"]}
    board.robber_coordinate = tuple(doc["robber_coordinate"])
    board.connected_components = defaultdict(
        list,
        {
            Color[c]: [set(s) for s in comps]
            for c, comps in doc["connected_components"].items()
        },
    )
    board.board_buildable_ids = set(doc["board_buildable_ids"])
    board.road_lengths = defaultdict(
        int, {Color[c]: n for c, n in doc["road_lengths"].items()}
    )
    board.road_color = Color[doc["road_color"]] if doc["road_color"] else None
    board.road_length = doc["road_length"]
    board.buildable_subgraph = STATIC_GRAPH.subgraph(catan_map.land_nodes)
    board.buildable_edges_cache = {}
    board.player_port_resources_cache = {}
    return board


# ===== actions =====
def action_to_json(a: Action):
    return [a.color.value, a.action_type.value, _plain(a.value)]


def _plain(v):
    if isinstance(v, Color):
        return v.value
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def action_from_json(payload) -> Action:
    """Decode ``[color, action_type, value]``.

    The one decoder, used for actions posted by the web UI and for actions
    read back out of a stored game.
    """
    color, action_type, value = Color[payload[0]], ActionType[payload[1]], payload[2]
    if action_type == ActionType.BUILD_ROAD:
        value = tuple(value)
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        value = tuple(value)
        if len(value) not in (1, 2):
            raise ValueError("Year of Plenty action must have 1 or 2 resources")
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        value = (tuple(coordinate), Color[victim] if victim else None)
    elif action_type == ActionType.CONFIRM_TRADE:
        value = tuple(value[:10]) + (Color[value[10]],)
    elif isinstance(value, list):
        value = tuple(value)
    return Action(color, action_type, value)


# ===== state =====
def state_to_json(game: Game):
    s = game.state
    return {
        "schema_version": SCHEMA_VERSION,
        "game": {
            "id": game.id,
            "seed": game.seed,
            "vps_to_win": game.vps_to_win,
            "discard_limit": s.discard_limit,
            "friendly_robber": s.friendly_robber,
        },
        "map": map_to_json(s.board.map),
        "board": board_to_json(s.board),
        "colors": [c.value for c in s.colors],
        "player_state": s.player_state,
        "buildings_by_color": {
            c.value: {bt: _plain(v) for bt, v in d.items()}
            for c, d in s.buildings_by_color.items()
        },
        "resource_freqdeck": list(s.resource_freqdeck),
        "development_listdeck": list(s.development_listdeck),  # HIDDEN INFO
        "action_records": [
            [action_to_json(r.action), _plain(r.result)] for r in s.action_records
        ],
        "num_turns": s.num_turns,
        "current_player_index": s.current_player_index,
        "current_turn_index": s.current_turn_index,
        "current_prompt": s.current_prompt.value,
        "is_initial_build_phase": s.is_initial_build_phase,
        "is_discarding": s.is_discarding,
        "discard_counts": list(s.discard_counts),
        "is_moving_knight": s.is_moving_knight,
        "is_road_building": s.is_road_building,
        "free_roads_available": s.free_roads_available,
        "is_resolving_trade": s.is_resolving_trade,
        "current_trade": list(s.current_trade),
        "acceptees": list(s.acceptees),
    }


def state_from_json(doc, players) -> Game:
    """Rebuild a Game from a document.

    ``players`` are built separately (from their specs, via the registry); the
    document describes the game, never the code that plays it.
    """
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {doc['schema_version']}")
    if not isinstance(doc["development_listdeck"], list):
        # The ordered deck is the tell: client_view() turns it into a count.
        raise ValueError(
            "this is a client_view, not the authoritative document; a redacted "
            "view is missing what a game needs to be rebuilt"
        )
    catan_map = map_from_json(doc["map"])
    s = State([], None, initialize=False)
    s.colors = tuple(Color[c] for c in doc["colors"])
    s.color_to_index = {c: i for i, c in enumerate(s.colors)}
    by_color = {p.color: p for p in players}
    if set(by_color) != set(s.colors):
        raise ValueError(
            f"players do not match the document: expected colors "
            f"{[c.value for c in s.colors]}, got "
            f"{sorted(c.value for c in by_color)}"
        )
    s.players = [by_color[c] for c in s.colors]  # seating order preserved
    s.discard_limit = doc["game"]["discard_limit"]
    s.friendly_robber = doc["game"]["friendly_robber"]
    s.board = board_from_json(doc["board"], catan_map)
    s.player_state = dict(doc["player_state"])
    s.resource_freqdeck = list(doc["resource_freqdeck"])
    s.development_listdeck = list(doc["development_listdeck"])
    s.buildings_by_color = {
        Color[c]: defaultdict(
            list,
            {
                bt: [tuple(x) if isinstance(x, list) else x for x in v]
                for bt, v in d.items()
            },
        )
        for c, d in doc["buildings_by_color"].items()
    }
    s.action_records = [
        ActionRecord(action_from_json(a), tuple(r) if isinstance(r, list) else r)
        for a, r in doc["action_records"]
    ]
    s.num_turns = doc["num_turns"]
    s.current_player_index = doc["current_player_index"]
    s.current_turn_index = doc["current_turn_index"]
    s.current_prompt = ActionPrompt(doc["current_prompt"])
    s.is_initial_build_phase = doc["is_initial_build_phase"]
    s.is_discarding = doc["is_discarding"]
    s.discard_counts = list(doc["discard_counts"])
    s.is_moving_knight = doc["is_moving_knight"]
    s.is_road_building = doc["is_road_building"]
    s.free_roads_available = doc["free_roads_available"]
    s.is_resolving_trade = doc["is_resolving_trade"]
    s.current_trade = tuple(doc["current_trade"])
    s.acceptees = tuple(doc["acceptees"])

    game = Game(players=[], initialize=False)
    game.id = doc["game"]["id"]
    game.seed = doc["game"]["seed"]
    game.vps_to_win = doc["game"]["vps_to_win"]
    game.friendly_robber = s.friendly_robber
    game.state = s
    game.playable_actions = generate_playable_actions(s)
    return game


#: Whether a seat held that card when its turn began: reveals the hand. Taken
#: from the state blueprint rather than DEVELOPMENT_CARDS, because only the
#: playable four have the flag.
HAND_FLAGS = tuple(k for k in PLAYER_INITIAL_STATE if k.endswith("_OWNED_AT_START"))
#: Which card you drew is yours alone. Where the robber went is public; what
#: it stole is not.
SECRET_VALUE = {"BUY_DEVELOPMENT_CARD"}
SECRET_RESULT = {"BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"}


def client_view(doc, perspective=None):
    """Redacted projection of a document, safe to send to a browser or a bot.

    Always removes what nobody at the table knows: the seed, and the order of
    the development deck. The deck's *composition* survives, because a player
    legitimately reasons about the odds of drawing a knight; only its order,
    which would reveal future draws, is removed.

    ``perspective`` additionally removes what only the *other* seats know --
    their hands, and the private half of what they did. Pass it for anyone
    who is playing (a bot on the wire always gets it); leave it out for a
    spectator, a replay, or an accumulator collecting training data.
    """
    view = dict(doc)
    view["development_listdeck"] = dict(
        sorted(Counter(doc["development_listdeck"]).items())
    )
    view["game"] = {k: v for k, v in doc["game"].items() if k != "seed"}
    if perspective is None:
        return view

    color = getattr(perspective, "value", perspective)
    if color not in doc["colors"]:
        raise ValueError(f"{color!r} is not seated in this game: {doc['colors']}")
    seat = doc["colors"].index(color)

    state = dict(doc["player_state"])
    for index in range(len(doc["colors"])):
        if index == seat:
            continue
        prefix = f"P{index}_"
        state[prefix + "NUM_RESOURCES_IN_HAND"] = sum(
            state.pop(f"{prefix}{resource}_IN_HAND") for resource in RESOURCES
        )
        state[prefix + "NUM_DEVELOPMENT_CARDS_IN_HAND"] = sum(
            state.pop(f"{prefix}{card}_IN_HAND") for card in DEVELOPMENT_CARDS
        )
        for flag in HAND_FLAGS:
            state.pop(prefix + flag)
        # Counts the victory-point cards still in hand.
        state.pop(prefix + "ACTUAL_VICTORY_POINTS")
    view["player_state"] = state

    view["action_records"] = [
        (
            [[a[0], a[1], None if a[1] in SECRET_VALUE else a[2]], None]
            if a[0] != color and a[1] in SECRET_RESULT
            else [a, result]
        )
        for a, result in doc["action_records"]
    ]
    return view


# ===== what a browser needs on top of the document =====
def _tile_json(tile):
    if isinstance(tile, Water):
        return {"type": "WATER"}
    if isinstance(tile, Port):
        return {
            "id": tile.id,
            "type": "PORT",
            "direction": tile.direction.value,
            "resource": tile.resource,
        }
    if tile.resource is None:
        return {"id": tile.id, "type": "DESERT"}
    return {
        "id": tile.id,
        "type": "RESOURCE_TILE",
        "resource": tile.resource,
        "number": tile.number,
    }


def geometry(game) -> dict:
    """Board layout: node and edge ids, and what sits on them.

    Derived from ``map``, which the document already carries; the browser
    would otherwise have to re-implement the topology walk in TypeScript.
    """
    board = game.state.board
    nodes, edges = {}, {}
    for coordinate, tile in board.map.tiles.items():
        for direction, node_id in tile.nodes.items():
            building = board.buildings.get(node_id, None)
            nodes[node_id] = {
                "id": node_id,
                "tile_coordinate": coordinate,
                "direction": direction.value,
                "building": None if building is None else building[1],
                "color": None if building is None else building[0].value,
            }
        for direction, edge in tile.edges.items():
            color = board.roads.get(edge, None)
            edges[tuple(sorted(edge))] = {
                "id": tuple(sorted(edge)),
                "tile_coordinate": coordinate,
                "direction": direction.value,
                "color": None if color is None else color.value,
            }
    return {
        "tiles": [
            {"coordinate": coordinate, "tile": _tile_json(tile)}
            for coordinate, tile in board.map.tiles.items()
        ],
        "adjacent_tiles": {
            node_id: [_tile_json(t) for t in tiles]
            for node_id, tiles in board.map.adjacent_tiles.items()
        },
        "nodes": nodes,
        "edges": list(edges.values()),
    }


def web_view(game, perspective=None) -> dict:
    """The single payload the web UI receives.

    The redacted state document, plus the few things a browser needs that are
    not state at all: the actions that are legal right now (engine output),
    which seats are bots (a property of the players), and the board geometry
    (derived from the map).

    ``perspective`` is passed through to :func:`client_view`: with it the
    browser sees the table as that seat does, without it as a spectator.
    """
    state = game.state
    winner = game.winning_color()
    view = client_view(state_to_json(game), perspective)
    view.update(geometry(game))
    view.update(
        {
            "current_color": state.current_color().value,
            "current_playable_actions": [
                action_to_json(a) for a in game.playable_actions
            ],
            "current_discard_count": state.discard_counts[state.current_player_index],
            "bot_colors": [p.color.value for p in state.players if p.is_bot],
            "winning_color": winner.value if winner else None,
            "state_index": len(state.action_records),
            "robber_coordinate": list(state.board.robber_coordinate),
            "longest_roads_by_player": {
                color.value: get_longest_road_length(state, color)
                for color in state.colors
            },
        }
    )
    return view
