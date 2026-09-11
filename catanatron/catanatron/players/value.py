from functools import cached_property
from dataclasses import dataclass
from typing import Literal, Optional


from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.player import Player
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
from catanatron.features import (
    build_production_features,
    reachability_features,
)

TRANSLATE_VARIETY = 4  # i.e. each new resource is like 4 production points

DEFAULT_WEIGHTS = {
    # Where to place. Note winning is best at all costs
    "public_vps": 3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    # Towards where to expand and when
    "reachable_production_0": 0,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10,
    # Hand, when to hold and when to use.
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
}

# Change these to play around with new values
CONTENDER_WEIGHTS = {
    "public_vps": 300000000000001.94,
    "production": 100000002.04188395,
    "enemy_production": -99999998.03389844,
    "num_tiles": 2.91440418,
    "reachable_production_0": 2.03820085,
    "reachable_production_1": 10002.018773150001,
    "buildable_nodes": 1001.86278466,
    "longest_road": 12.127388499999999,
    "hand_synergy": 102.40606877,
    "hand_resources": 2.43644327,
    "discard_penalty": -3.00141993,
    "hand_devs": 10.721669799999999,
    "army_size": 12.93844622,
}


def base_fn(params=DEFAULT_WEIGHTS, *, cache=None):
    def fn(game, p0_color):
        (
            production,
            enemy_production,
            reachable_production_at_zero,
            reachable_production_at_one,
            num_tiles,
            num_buildable_nodes,
        ) = (
            cache.terms(game, p0_color)
            if cache is not None
            else _board_terms(game, p0_color)
        )
        key = player_key(game.state, p0_color)
        longest_road_length = get_longest_road_length(game.state, p0_color)

        player_state = game.state.player_state
        wheat = player_state[key + "_WHEAT_IN_HAND"]
        ore = player_state[key + "_ORE_IN_HAND"]
        sheep = player_state[key + "_SHEEP_IN_HAND"]
        brick = player_state[key + "_BRICK_IN_HAND"]
        wood = player_state[key + "_WOOD_IN_HAND"]
        distance_to_city = (
            max(2 - wheat, 0) + max(3 - ore, 0)
        ) / 5.0  # 0 means good. 1 means bad.
        distance_to_settlement = (
            max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)
        ) / 4.0  # 0 means good. 1 means bad.
        hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

        num_in_hand = player_num_resource_cards(game.state, p0_color)
        discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

        longest_road_factor = (
            params["longest_road"] if num_buildable_nodes == 0 else 0.1
        )

        return float(
            game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
            + production * params["production"]
            + enemy_production * params["enemy_production"]
            + reachable_production_at_zero * params["reachable_production_0"]
            + reachable_production_at_one * params["reachable_production_1"]
            + hand_synergy * params["hand_synergy"]
            + num_buildable_nodes * params["buildable_nodes"]
            + num_tiles * params["num_tiles"]
            + num_in_hand * params["hand_resources"]
            + discard_penalty
            + longest_road_length * longest_road_factor
            + player_num_dev_cards(game.state, p0_color) * params["hand_devs"]
            + get_played_dev_cards(game.state, p0_color, "KNIGHT") * params["army_size"]
        )

    return fn


def value_production(sample, player_name="P0", include_variety=True):
    proba_point = 2.778 / 100
    features = [
        f"EFFECTIVE_{player_name}_WHEAT_PRODUCTION",
        f"EFFECTIVE_{player_name}_ORE_PRODUCTION",
        f"EFFECTIVE_{player_name}_SHEEP_PRODUCTION",
        f"EFFECTIVE_{player_name}_WOOD_PRODUCTION",
        f"EFFECTIVE_{player_name}_BRICK_PRODUCTION",
    ]
    prod_sum = sum([sample[f] for f in features])
    prod_variety = (
        sum([sample[f] != 0 for f in features]) * TRANSLATE_VARIETY * proba_point
    )
    return prod_sum + (0 if not include_variety else prod_variety)


_production_features = build_production_features(True)


def _board_terms(game, color):
    # Native base_fn computes this same production sample twice.
    sample = _production_features(game, color)
    production = value_production(sample, "P0")
    enemy_production = value_production(sample, "P1", False)
    reach = reachability_features(game, color, 1, only_p0=True)
    zero = sum([reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES])
    one = sum([reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES])
    buildings = game.state.buildings_by_color[color]
    tiles = set()
    for node in buildings[SETTLEMENT] + buildings[CITY]:
        tiles.update(game.state.board.map.adjacent_tiles[node])
    return (
        production,
        enemy_production,
        zero,
        one,
        len(tiles),
        len(game.state.board.buildable_node_ids(color)),
    )


class BoardFeatureCache:
    """Bounded cache owned by one player and reused across its decisions.

    Map objects are retained and a map change clears entries. Ordered building
    lists preserve floating-point summation order. Component node iteration is
    retained because native reachability unions sets before summing production.
    Hands, VP, army, dev cards and longest-road length are read live, outside
    this cache. No game objects, final scores or search bounds are retained.
    """

    def __init__(self, max_entries=4096):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.colors = None
        self.map = None
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def terms(self, game, color):
        state = game.state
        board = state.board
        if board.map is not self.map or state.colors != self.colors:
            self.entries.clear()
            self.map = board.map
            self.colors = state.colors
        # Cache the inputs actually consumed by the board terms. Enemy
        # ownership identity/type is irrelevant to P0 reachability; its
        # node/edge blockers are sufficient. Ordered per-seat buildings
        # still identify production for every color and relative P1.
        key = (
            color.value,
            board.robber_coordinate,
            tuple(
                (
                    tuple(state.buildings_by_color[c][SETTLEMENT]),
                    tuple(state.buildings_by_color[c][CITY]),
                )
                for c in self.colors
            ),
            frozenset(
                n
                for n, owner in board.buildings.items()
                if owner is not None and owner[0] != color
            ),
            frozenset(
                e
                for e, owner in board.roads.items()
                if owner is not None and owner != color
            ),
            tuple(tuple(component) for component in board.connected_components[color]),
            tuple(board.board_buildable_ids),
        )
        value = self.entries.get(key)
        if value is not None:
            self.hits += 1
            return value
        self.misses += 1
        value = _board_terms(game, color)
        if len(self.entries) >= self.max_entries:
            # FIFO bounds memory without updating an LRU chain at every leaf.
            del self.entries[next(iter(self.entries))]
        self.entries[key] = value
        return value

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "max_entries": self.max_entries,
        }


def contender_fn(params, *, cache=None):
    return base_fn(params or CONTENDER_WEIGHTS, cache=cache)


class ValueFunctionPlayer(Player):
    """
    Player that selects the move that maximizes a heuristic value function.

    For now, the base value function only considers 1 enemy player.
    """

    LABEL = "Value Function"

    @dataclass(frozen=True)
    class Params:
        value_fn: Literal["base", "contender"] = "base"
        epsilon: Optional[float] = None
        #: Non-scalar: programmatic use only, not settable from CLI/web.
        weights: Optional[dict] = None

    @property
    def value_fn_builder_name(self):
        return "contender_fn" if self.params.value_fn == "contender" else "base_fn"

    @cached_property
    def _board_feature_cache(self):
        return BoardFeatureCache()

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        epsilon = self.params.epsilon
        if epsilon is not None and game.state.random.random() < epsilon:
            return game.state.random.choice(playable_actions)

        best_value = float("-inf")
        best_action = None
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params.weights,
                cache=self._board_feature_cache,
            )
            value = value_fn(game_copy, self.color)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action


def get_value_fn(name, params, value_function=None, *, cache=None):
    if value_function is not None:
        return value_function
    elif name == "base_fn":
        return base_fn(DEFAULT_WEIGHTS, cache=cache)
    elif name == "contender_fn":
        return contender_fn(params, cache=cache)
    else:
        raise ValueError
