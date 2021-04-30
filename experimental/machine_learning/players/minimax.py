import random
import time
from collections import defaultdict
from typing import List

from catanatron.models.enums import DevelopmentCard, Resource
from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.actions import Action
from catanatron.models.enums import Resource, DevelopmentCard
from experimental.machine_learning.features import (
    build_production_features,
    create_sample,
    iter_players,
)
from experimental.machine_learning.players.tree_search_utils import (
    execute_spectrum,
    expand_spectrum,
)


effective_production_features = build_production_features(True)
total_production_features = build_production_features(False)


def build_value_function(params):
    def fn(game, p0_color):
        iterator = iter_players(game, p0_color)
        _, p0 = next(iterator)

        sample = create_sample(game, p0_color)

        proba_point = 2.778 / 100
        features = [
            "EFFECTIVE_P0_WHEAT_PRODUCTION",
            "EFFECTIVE_P0_ORE_PRODUCTION",
            "EFFECTIVE_P0_SHEEP_PRODUCTION",
            "EFFECTIVE_P0_WOOD_PRODUCTION",
            "EFFECTIVE_P0_BRICK_PRODUCTION",
        ]
        prod_sum = sum([sample[f] for f in features])
        prod_variety = sum([sample[f] != 0 for f in features]) * 4 * proba_point

        enemy_features = [
            "EFFECTIVE_P1_WHEAT_PRODUCTION",
            "EFFECTIVE_P1_ORE_PRODUCTION",
            "EFFECTIVE_P1_SHEEP_PRODUCTION",
            "EFFECTIVE_P1_WOOD_PRODUCTION",
            "EFFECTIVE_P1_BRICK_PRODUCTION",
        ]
        enemy_prod_sum = sum([sample[f] for f in enemy_features])

        longest_road_length = sample["P0_LONGEST_ROAD_LENGTH"]

        features = [f"P0_1_ROAD_REACHABLE_{resource.value}" for resource in Resource]
        production_at_one = sum([sample[f] for f in features])

        return float(
            p0.public_victory_points * 1000000
            + longest_road_length * params[0]
            + len(p0.development_deck.to_array()) * 10
            + len(p0.played_development_cards.to_array()) * 10.1
            + p0.played_development_cards.count(DevelopmentCard.KNIGHT) * 10.1
            + prod_sum * 1000
            + prod_variety * 1000
            - enemy_prod_sum * 1000
            + production_at_one * 10
        )

    return fn


def get_value_fn_builder(name):
    return globals()[name]


class ValueFunctionPlayer(Player):
    def __init__(
        self, color, name, value_fn_builder_name="build_value_function", params=[4]
    ):
        super().__init__(color, name=name)
        self.value_fn_builder_name = value_fn_builder_name
        self.params = params

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_action = None
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            value_fn = get_value_fn_builder(self.value_fn_builder_name)(self.params)
            value = value_fn(game_copy, self.color)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def __str__(self) -> str:
        return super().__str__() + self.value_fn_builder_name + str(self.params)


class VictoryPointPlayer(Player):
    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            value = game_copy.state.players_by_color[self.color].actual_victory_points
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_value = value
                best_actions = [action]

        return random.choice(best_actions)


ALPHABETA_DEFAULT_DEPTH = 2
MAX_SEARCH_TIME_SECS = 20


class AlphaBetaPlayer(Player):
    def __init__(
        self,
        color,
        name,
        depth=ALPHABETA_DEFAULT_DEPTH,
        value_fn_builder_name="build_value_function",
        params=[4],
    ):
        super().__init__(color, name=name)
        self.depth = depth
        self.value_fn_builder_name = value_fn_builder_name
        self.params = params

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        value_fn = get_value_fn_builder(self.value_fn_builder_name)(self.params)
        start = time.time()
        deadline = start + MAX_SEARCH_TIME_SECS
        result = alphabeta(
            game.copy(),
            self.depth,
            float("-inf"),
            float("inf"),
            self.color,
            deadline,
            value_fn,
        )
        print(
            "Decision Results:", self.depth, len(playable_actions), time.time() - start
        )
        # breakpoint()
        return result[0]

    def __repr__(self) -> str:
        return super().__repr__() + f"(depth={self.depth})"


def alphabeta(game, depth, alpha, beta, p0_color, deadline, value_fn):
    """AlphaBeta MiniMax Algorithm.

    NOTE: Sometimes returns a value, sometimes an (action, value). This is
    because some levels are state=>action, some are action=>state and in
    action=>state would probably need (action, proba, value) as return type.
    """
    # tabs = "\t" * (ALPHABETA_DEFAULT_DEPTH - depth)
    if depth == 0 or game.winning_color() is not None or time.time() >= deadline:
        # print(tabs, "returned heuristic", value_fn(game, p0_color))
        return value_fn(game, p0_color)

    maximizingPlayer = game.state.current_player().color == p0_color
    actions = game.state.playable_actions
    children = expand_spectrum(game, actions)
    # print(tabs, "MAXIMIZING =", maximizingPlayer, len(children))
    if maximizingPlayer:
        best_action = None
        best_value = float("-inf")
        for action, outprobas in children.items():
            expected_value = 0
            for (out, proba) in outprobas:
                # print(tabs, "call maxalphabeta", action, proba, depth - 1, alpha, beta)
                result = alphabeta(
                    out, depth - 1, alpha, beta, p0_color, deadline, value_fn
                )
                value = result if isinstance(result, float) else result[1]
                expected_value += proba * value

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                # print(tabs, "beta cutoff")
                break  # beta cutoff
            # print(tabs, "Expected Value:", action, expected_value, alpha, beta)

        return best_action, best_value
    else:
        best_action = None
        best_value = float("inf")
        for action, outprobas in children.items():
            expected_value = 0
            for (out, proba) in outprobas:
                # print(tabs, "call minalphabeta", action, proba, depth - 1, alpha, beta)
                result = alphabeta(
                    out, depth - 1, alpha, beta, p0_color, deadline, value_fn
                )
                value = result if isinstance(result, float) else result[1]
                expected_value += proba * value

            if expected_value < best_value:
                best_action = action
                best_value = expected_value
            beta = min(beta, best_value)
            if beta <= alpha:
                # print(tabs, "alpha cutoff")
                break  # alpha cutoff
            # print(tabs, "Expected Value:", action, expected_value, alpha, beta)

        return best_action, best_value
