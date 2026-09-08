"""Tests that JSON alone completely defines a game.

These are what make it safe to persist games without pickle: if a document
round-trips exactly, and a game resumed from it plays out identically, then
nothing about the game lives outside the document.
"""

import json
import random

import pytest

from catanatron import Game
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players import register_builtins
from catanatron.registry import REGISTRY
from catanatron.serialization import (
    SCHEMA_VERSION,
    client_view,
    detect_template,
    state_from_json,
    state_to_json,
)

register_builtins()


@pytest.fixture(autouse=True)
def preserve_random_state():
    """Keep these tests' seeding from leaking into the rest of the suite."""
    state = random.getstate()
    yield
    random.setstate(state)


#: Fields that fully describe a State; mirrors State.copy().
STATE_FIELDS = [
    "colors",
    "player_state",
    "resource_freqdeck",
    "development_listdeck",
    "num_turns",
    "current_player_index",
    "current_turn_index",
    "current_prompt",
    "is_initial_build_phase",
    "is_discarding",
    "discard_counts",
    "is_moving_knight",
    "is_road_building",
    "free_roads_available",
    "is_resolving_trade",
    "current_trade",
    "acceptees",
    "discard_limit",
    "friendly_robber",
]
BOARD_FIELDS = ["buildings", "roads", "robber_coordinate", "road_color", "road_length"]


def played_game(ticks=120, seed=7, map_type="BASE", players=None):
    random.seed(seed)  # callers restore global RNG state via the autouse fixture
    players = players or [RandomPlayer(color) for color in Color]
    game = Game(players, catan_map=build_map(map_type))
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game, players


def round_trip(game, players):
    """Serialize through real JSON text, then hydrate."""
    doc = json.loads(json.dumps(state_to_json(game)))
    return state_from_json(doc, players), doc


def assert_states_equal(a, b):
    for field in STATE_FIELDS:
        left, right = getattr(a, field), getattr(b, field)
        if isinstance(left, (list, tuple)):
            assert list(left) == list(right), field
        else:
            assert left == right, field
    for field in BOARD_FIELDS:
        assert getattr(a.board, field) == getattr(b.board, field), field
    assert dict(a.board.road_lengths) == dict(b.board.road_lengths)
    assert a.board.board_buildable_ids == b.board.board_buildable_ids
    assert {
        c: [set(s) for s in v] for c, v in a.board.connected_components.items()
    } == {c: [set(s) for s in v] for c, v in b.board.connected_components.items()}
    assert {c: dict(d) for c, d in a.buildings_by_color.items()} == {
        c: dict(d) for c, d in b.buildings_by_color.items()
    }
    assert a.action_records == b.action_records


def test_document_is_json_serializable():
    game, _ = played_game()
    assert isinstance(json.dumps(state_to_json(game)), str)


def test_document_carries_schema_version():
    game, _ = played_game()
    assert state_to_json(game)["schema_version"] == SCHEMA_VERSION


def test_unknown_schema_version_is_rejected():
    game, players = played_game(ticks=10)
    doc = state_to_json(game)
    doc["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="unsupported schema_version"):
        state_from_json(doc, players)


@pytest.mark.parametrize("ticks", [0, 8, 60, 120])
def test_round_trip_preserves_every_state_field(ticks):
    game, players = played_game(ticks=ticks)
    hydrated, _ = round_trip(game, players)
    assert_states_equal(game.state, hydrated.state)


def test_round_trip_preserves_playable_actions():
    game, players = played_game()
    hydrated, _ = round_trip(game, players)
    assert sorted(map(str, game.playable_actions)) == sorted(
        map(str, hydrated.playable_actions)
    )


def test_round_trip_preserves_game_config():
    random.seed(1)
    players = [RandomPlayer(color) for color in Color]
    game = Game(players, discard_limit=5, vps_to_win=8, friendly_robber=True)
    for _ in range(30):
        game.play_tick()
    hydrated, _ = round_trip(game, players)
    assert hydrated.vps_to_win == 8
    assert hydrated.state.discard_limit == 5
    assert hydrated.state.friendly_robber is True
    assert hydrated.id == game.id
    assert hydrated.seed == game.seed


@pytest.mark.parametrize("map_type", ["BASE", "MINI", "TOURNAMENT"])
def test_round_trip_preserves_map(map_type):
    game, players = played_game(ticks=60, map_type=map_type)
    hydrated, _ = round_trip(game, players)
    original, rebuilt = game.state.board.map, hydrated.state.board.map
    assert {
        c: (getattr(t, "resource", None), getattr(t, "number", None))
        for c, t in original.tiles.items()
    } == {
        c: (getattr(t, "resource", None), getattr(t, "number", None))
        for c, t in rebuilt.tiles.items()
    }
    assert original.port_nodes == rebuilt.port_nodes
    assert original.node_production == rebuilt.node_production
    assert original.land_nodes == rebuilt.land_nodes


def test_tournament_map_round_trips_as_base_topology():
    game, _ = played_game(ticks=5, map_type="TOURNAMENT")
    assert detect_template(game.state.board.map) == "BASE"


def test_hydrated_game_plays_out_identically():
    """The strongest guarantee: same RNG in, same game out."""
    game, players = played_game(ticks=120)
    assert game.winning_color() is None, "fixture must still be mid-game"
    hydrated, doc = round_trip(game, players)

    rng_state = random.getstate()
    original_winner = game.play()
    random.setstate(rng_state)
    hydrated_winner = hydrated.play()

    assert original_winner == hydrated_winner
    assert game.state.action_records == hydrated.state.action_records
    assert game.state.player_state == hydrated.state.player_state


def test_round_trip_with_parameterized_bots():
    players = [
        AlphaBetaPlayer(Color.RED, AlphaBetaPlayer.Params(depth=2, prunning=True)),
        RandomPlayer(Color.BLUE),
    ]
    game, players = played_game(ticks=40, players=players)
    hydrated, _ = round_trip(game, players)
    assert_states_equal(game.state, hydrated.state)


def test_players_come_from_specs_not_from_the_document():
    """A document describes a game, never the code that plays it."""
    game, _ = played_game(ticks=30)
    doc = json.loads(json.dumps(state_to_json(game)))
    assert "players" not in doc

    specs = ["R", "W", "AB:depth=2", "VP"]
    rebuilt_players = REGISTRY.build_all(specs, colors=game.state.colors)
    hydrated = state_from_json(doc, rebuilt_players)
    assert [type(p).__name__ for p in hydrated.state.players] == [
        "RandomPlayer",
        "WeightedRandomPlayer",
        "AlphaBetaPlayer",
        "VictoryPointPlayer",
    ]
    assert hydrated.state.colors == game.state.colors


# ===== redaction =====
def test_client_view_hides_deck_order_but_keeps_composition():
    game, _ = played_game()
    doc = state_to_json(game)
    view = client_view(doc)

    assert isinstance(doc["development_listdeck"], list), "authoritative doc is ordered"
    assert isinstance(view["development_listdeck"], dict), "client view is a count"
    assert sum(view["development_listdeck"].values()) == len(
        doc["development_listdeck"]
    )
    for card in set(doc["development_listdeck"]):
        assert view["development_listdeck"][card] == doc["development_listdeck"].count(
            card
        )


def test_client_view_hides_the_seed():
    game, _ = played_game()
    view = client_view(state_to_json(game))
    assert "seed" not in view["game"]


def test_client_view_does_not_mutate_the_document():
    game, _ = played_game()
    doc = state_to_json(game)
    before = json.dumps(doc)
    client_view(doc)
    assert json.dumps(doc) == before


# ===== the honest view: what one seat is entitled to see =====
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVCARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")


def seats(game, color=Color.RED):
    """(doc, honest view, my prefix, an opponent's prefix).

    Seating order is shuffled, so P0 is not necessarily RED.
    """
    doc = state_to_json(game)
    mine = doc["colors"].index(color.value)
    theirs = (mine + 1) % len(doc["colors"])
    return doc, client_view(doc, color), f"P{mine}_", f"P{theirs}_"


def test_my_own_hand_stays_itemized():
    game, _ = played_game()
    _, view, mine, _ = seats(game)
    for resource in RESOURCES:
        assert mine + resource + "_IN_HAND" in view["player_state"]
    assert mine + "ACTUAL_VICTORY_POINTS" in view["player_state"]


def test_an_opponents_hand_becomes_a_count():
    """Across the table you can count someone's cards, not read them."""
    game, _ = played_game()
    doc, view, _, theirs = seats(game)
    state, truth = view["player_state"], doc["player_state"]

    assert not [
        key
        for key in state
        if key.startswith(theirs)
        and key.endswith("_IN_HAND")
        and not key.startswith(theirs + "NUM_")
    ]
    assert state[theirs + "NUM_RESOURCES_IN_HAND"] == sum(
        truth[f"{theirs}{resource}_IN_HAND"] for resource in RESOURCES
    )
    assert state[theirs + "NUM_DEVELOPMENT_CARDS_IN_HAND"] == sum(
        truth[f"{theirs}{card}_IN_HAND"] for card in DEVCARDS
    )


def test_an_opponents_hidden_victory_points_are_not_published():
    game, _ = played_game()
    _, view, _, theirs = seats(game)
    state = view["player_state"]
    assert theirs + "ACTUAL_VICTORY_POINTS" not in state
    assert theirs + "VICTORY_POINTS" in state, "public victory points stay public"
    assert theirs + "KNIGHT_OWNED_AT_START" not in state
    assert theirs + "PLAYED_KNIGHT" in state, "what was played is public"


def test_the_card_an_opponent_drew_is_not_in_the_history():
    game, _ = played_game(ticks=400)
    _, view, _, _ = seats(game)
    bought = [
        (action, result)
        for action, result in view["action_records"]
        if action[1] == "BUY_DEVELOPMENT_CARD"
    ]
    assert bought, "the fixture should have bought development cards"
    for action, result in bought:
        if action[0] == Color.RED.value:
            assert action[2] is not None, "I know what I drew"
        else:
            assert (action[2], result) == (None, None)


def test_what_the_robber_stole_is_hidden_but_where_it_went_is_not():
    game, _ = played_game(ticks=400)
    _, view, _, _ = seats(game)
    moves = [
        (action, result)
        for action, result in view["action_records"]
        if action[1] == "MOVE_ROBBER" and action[0] != Color.RED.value
    ]
    assert moves, "the fixture should have moved the robber"
    for action, result in moves:
        coordinate, victim = action[2]
        assert isinstance(coordinate, list), "the tile is public"
        assert victim is None or isinstance(victim, str), "who was robbed is public"
        assert result is None, "which resource is not"


def test_a_spectator_still_sees_everything():
    game, _ = played_game()
    view = client_view(state_to_json(game))
    assert "P1_WOOD_IN_HAND" in view["player_state"]
    assert "P1_NUM_RESOURCES_IN_HAND" not in view["player_state"]


def test_perspective_accepts_a_color_or_its_name():
    game, _ = played_game()
    doc = state_to_json(game)
    assert client_view(doc, Color.RED) == client_view(doc, "RED")


def test_a_perspective_that_is_not_seated_is_rejected():
    game, _ = played_game(players=[RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    with pytest.raises(ValueError, match="not seated in this game"):
        client_view(state_to_json(game), Color.WHITE)


def test_the_honest_view_does_not_mutate_the_document():
    game, _ = played_game()
    doc = state_to_json(game)
    before = json.dumps(doc)
    client_view(doc, Color.RED)
    assert json.dumps(doc) == before


def test_a_redacted_view_cannot_rebuild_a_game():
    """Hydrating from what the browser was sent would silently lose the deck."""
    game, players = played_game(ticks=10)
    for view in (
        client_view(state_to_json(game)),
        client_view(state_to_json(game), Color.RED),
    ):
        with pytest.raises(ValueError, match="not the authoritative document"):
            state_from_json(view, players)


def test_mismatched_players_are_rejected():
    game, _ = played_game(ticks=10)
    doc = state_to_json(game)
    with pytest.raises(ValueError, match="players do not match the document"):
        state_from_json(doc, [RandomPlayer(Color.RED)])
