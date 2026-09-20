"""Exact evaluator output and mutable-copy contracts for the speed changes."""

import pickle
import random
from collections import defaultdict

import pytest

from catanatron.game import Game
from catanatron.models.enums import SETTLEMENT, CITY
from catanatron.models.player import Color, RandomPlayer
from catanatron.features import reachability_features
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS, BoardFeatureCache
from tests.value_reference import reference_base_fn


def position(seed=17):
    return Game([RandomPlayer(c) for c in Color], seed=seed)


@pytest.mark.parametrize("cached", [False, True])
def test_exact_scalar_for_every_perspective_and_nondefault_weights(cached):
    varied = {key: (i + 1) / 7 for i, key in enumerate(DEFAULT_WEIGHTS)}
    cache = BoardFeatureCache() if cached else None
    evaluators = [
        (base_fn(p, cache=cache), reference_base_fn(p))
        for p in (DEFAULT_WEIGHTS, varied)
    ]
    for seed in (7, 19):
        game = position(seed)
        rng = random.Random(seed)
        for tick in range(300):
            if tick % 7 == 0:
                for color in game.state.colors:
                    for actual, reference in evaluators:
                        assert actual(game, color) == reference(game, color)
                    full = reachability_features(game, color, 2)
                    subset = reachability_features(game, color, 1, only_p0=True)
                    assert len(subset) == 10
                    assert all(full[k] == v for k, v in subset.items())
            if game.winning_color() is not None:
                break
            game.execute(rng.choice(game.playable_actions))


def test_board_copy_matches_nested_container_values_order_and_isolation():
    board = position().state.board
    color = Color.RED
    board.connected_components[color] = [{0, 1}, {3, 4}]
    board.buildable_edges_cache[color] = [(0, 1), (3, 4)]
    board.player_port_resources_cache[color] = {"WOOD", "ORE"}
    original = pickle.loads(pickle.dumps(board.connected_components))
    copied = board.copy()
    assert vars(copied) == vars(board)
    assert isinstance(copied.connected_components, defaultdict)
    assert (
        copied.connected_components.default_factory
        is board.connected_components.default_factory
    )
    assert [list(c) for c in copied.connected_components[color]] == [
        list(c) for c in original[color]
    ]
    for name in ("map", "buildable_subgraph"):
        assert getattr(copied, name) is getattr(board, name)
    for name in (
        "buildings",
        "roads",
        "connected_components",
        "board_buildable_ids",
        "road_lengths",
        "buildable_edges_cache",
        "player_port_resources_cache",
    ):
        assert getattr(copied, name) is not getattr(board, name)
    copied.connected_components[color][0].add(9)
    copied.buildable_edges_cache[color].append((8, 9))
    copied.player_port_resources_cache[color].add("SHEEP")
    assert board.connected_components[color][0] == {0, 1}
    assert board.buildable_edges_cache[color] == [(0, 1), (3, 4)]
    assert board.player_port_resources_cache[color] == {"WOOD", "ORE"}


def test_state_copy_preserves_rng_and_nested_building_contract():
    state = position().state
    color = state.colors[0]
    state.buildings_by_color[color][SETTLEMENT].extend([1, 3])
    state.buildings_by_color[color][CITY].append(5)
    expected = pickle.loads(pickle.dumps(state.buildings_by_color))
    copied = state.copy()
    assert set(vars(copied)) == set(vars(state))
    for name in vars(state):
        if name != "board":
            assert getattr(copied, name) == getattr(state, name)
    assert copied.random is state.random
    assert copied.players is state.players
    assert copied.colors is state.colors
    assert copied.buildings_by_color == expected
    for c, buildings in state.buildings_by_color.items():
        assert copied.buildings_by_color[c] is not buildings
        assert isinstance(copied.buildings_by_color[c], defaultdict)
        assert copied.buildings_by_color[c].default_factory is buildings.default_factory
        for kind, nodes in buildings.items():
            assert copied.buildings_by_color[c][kind] is not nodes
    copied.buildings_by_color[color][SETTLEMENT].append(7)
    assert state.buildings_by_color == expected


def test_cache_reads_dynamic_hand_points_and_longest_road_live():
    game = position()
    cache = BoardFeatureCache()
    fast = base_fn(cache=cache)
    native = reference_base_fn()
    color = game.state.colors[0]
    first = fast(game, color)
    for suffix, delta in (
        ("WHEAT_IN_HAND", 8),
        ("VICTORY_POINTS", 1),
        ("LONGEST_ROAD_LENGTH", 2),
        ("KNIGHT_IN_HAND", 1),
        ("PLAYED_KNIGHT", 1),
    ):
        key = "P0_" + suffix
        game.state.player_state[key] += delta
        assert fast(game, color) == native(game, color)
    assert fast(game, color) != first
    assert cache.misses == 1
    assert cache.hits > 1


def test_cache_invalidates_board_dependencies_and_bounds_memory():
    game = position()
    cache = BoardFeatureCache(max_entries=2)
    color = game.state.colors[0]
    cache.terms(game, color)
    board = game.state.board
    # Independent mutable dependencies, even when the resulting synthetic
    # state has redundant representations that are not mutually consistent.
    mutations = (
        lambda: setattr(
            board,
            "robber_coordinate",
            next(c for c in board.map.land_tiles if c != board.robber_coordinate),
        ),
        lambda: game.state.buildings_by_color[color][SETTLEMENT].append(0),
        lambda: game.state.buildings_by_color[color][CITY].append(1),
        lambda: board.buildings.update({0: (game.state.colors[1], SETTLEMENT)}),
        lambda: board.roads.update({(0, 1): game.state.colors[1]}),
        lambda: board.connected_components[color].append({0, 1}),
        lambda: board.board_buildable_ids.discard(0),
    )
    for mutate in mutations:
        before = cache.misses
        mutate()
        cached = cache.terms(game, color)
        assert cache.misses == before + 1
        assert cached == BoardFeatureCache().terms(game, color)
        assert len(cache.entries) <= 2
    other = position(18)
    cache.terms(other, other.state.colors[0])
    assert cache.map is other.state.board.map
    assert len(cache.entries) == 1


def test_player_caches_are_isolated_and_follow_game_changes():
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanatron.players.value import ValueFunctionPlayer

    first = AlphaBetaPlayer(Color.RED)
    second = ValueFunctionPlayer(Color.BLUE)
    assert first._board_feature_cache is not second._board_feature_cache
    assert first._board_feature_cache is first._board_feature_cache
    for seed in (1, 2):
        game = position(seed)
        first._board_feature_cache.terms(game, Color.RED)
        assert first._board_feature_cache.map is game.state.board.map
        assert len(first._board_feature_cache.entries) == 1
