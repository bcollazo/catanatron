import random
import time
from collections import defaultdict
from typing import List

import numpy as np

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.actions import Action
from catanatron.models.enums import Resource, DevelopmentCard
from experimental.machine_learning.features import (
    build_production_features,
    create_sample,
    iter_players,
)


effective_production_features = build_production_features(True)
total_production_features = build_production_features(False)


def value_fn(game, p0_color, verbose=False):
    iterator = iter_players(game, p0_color)
    _, p0 = next(iterator)

    proba_point = 2.778 / 100
    production = effective_production_features(game, p0_color)
    features = [
        "EFFECTIVE_P0_WHEAT_PRODUCTION",
        "EFFECTIVE_P0_ORE_PRODUCTION",
        "EFFECTIVE_P0_SHEEP_PRODUCTION",
        "EFFECTIVE_P0_WOOD_PRODUCTION",
        "EFFECTIVE_P0_BRICK_PRODUCTION",
    ]
    prod_sum = sum([production[f] for f in features])
    prod_variety = sum([production[f] != 0 for f in features]) * 4 * proba_point

    paths = game.state.board.continuous_roads_by_player(p0_color)
    path_lengths = map(lambda path: len(path), paths)
    longest_road_length = 0 if len(paths) == 0 else max(path_lengths)

    if verbose:
        print(prod_sum, prod_variety)

    return (
        p0.public_victory_points * 1000
        + p0.cities_available * -100
        + p0.settlements_available * -10
        + p0.roads_available * -1
        + longest_road_length
        + len(p0.development_deck.to_array())
        + len(p0.played_development_cards.to_array()) * 0.1
        + prod_sum
        + prod_variety
    )


def build_value_function(params):
    def fn(game, p0_color, verbose=False):
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
        prod_variety = sum([sample[f] != 0 for f in features]) * params[0] * proba_point

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

        if verbose:
            print(prod_sum, prod_variety)

        return float(
            p0.public_victory_points * 1000000
            + longest_road_length * 10
            + len(p0.development_deck.to_array()) * 10
            + len(p0.played_development_cards.to_array()) * 10.1
            + p0.played_development_cards.count(DevelopmentCard.KNIGHT) * 10.1
            + prod_sum * 1000
            + prod_variety * 1000
            - enemy_prod_sum * 1000
            + production_at_one * 10
        )

    return fn


value_fn2 = build_value_function([4])

def get_value_fn(name):
    return globals()[name or "value_fn2"]
    


class ValueFunctionPlayer(Player):
    def __init__(self, color, name, value_fn_name):
        super().__init__(color, name=name)
        self.value_fn_name = value_fn_name
        
    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_action = None
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            value = get_value_fn(self.value_fn_name)(game_copy, self.color)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def __str__(self) -> str:
        return super().__str__() + self.value_fn.__name__


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


class StateNode:
    def __init__(self, parent: "ActionNode", state: Game, children: List["ActionNode"]):
        self.parent = parent
        self.children = children  # are these ActionNodes?
        self.state = state  # tick q will be consumed after .expand()
        self.value = None  # min(children) or max(children)

    def expand(self):
        player, action_prompt = self.state.pop_from_queue()
        # TODO: Need to avoid knowing player's actions.
        actions = self.state.playable_actions(player, action_prompt)
        return list(map(lambda a: ActionNode(self, a), actions))


class ActionNode:
    def __init__(self, parent: StateNode, action: Action):
        self.parent = parent
        self.action = action

        self.value = None  # should be expected value of children
        self.children = None  # List[StateNode]
        self.children_probas = None  # List[float] aligned to self.children

    def expand(self):
        # [(game_copy, proba)]
        results = self.parent.state.execute_spectrum(self.action)
        return list(map(lambda x: (StateNode(self, x[0], None), x[1]), results))


class MiniMaxPlayer(Player):
    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Expand a layer. Ensure if its a max or min layer.
        #   For each action, explode action into group of results.
        # Expand many layers in the future. Save tree.
        # Traverse back starting from leafs.
        # For each level, if max layer
        # For each leaf, assign value.
        # Traverse up a level. If enemys turn, then min layer.
        #   min will take the value of all its children

        # Create tree.
        # For each leaf (will be state). Check Value function.
        # keep going up levels. aggregating on expected value if going to a A-level
        # if going to a S level, use min-or-max step.
        start = time.time()
        MAX_LEVELS = 3  # 0:root-State(max), 1:actions, 2:states, 3:actions, 4:state.

        levels = defaultdict(list)
        state = StateNode(None, game, playable_actions)
        levels[0].append(state)
        for action in playable_actions:
            levels[1].append(ActionNode(state, action))

        for i in range(2, MAX_LEVELS):
            parents = levels[i - 1]
            for node in parents:
                if isinstance(node, StateNode):
                    node.children = node.expand()
                    levels[i].extend(node.children)
                else:
                    node.children = node.expand()
                    levels[i].extend(list(map(lambda t: t[0], node.children)))

        for leaf_node in levels[MAX_LEVELS - 1]:
            leaf_node.value = value_fn(leaf_node.state, self.color)

        for level in range(MAX_LEVELS - 2, 0, -1):
            for node in levels[level]:  # evaluate
                if isinstance(node, StateNode):
                    max_level = node.state.current_player().color == self.color
                    fn = max if max_level else min
                    node.value = fn(map(lambda a: a.value, node.children))
                else:
                    node.value = sum(map(lambda s: s[0].value * s[1], node.children))

        best_action = max(levels[1], key=lambda a: a.value).action
        # level_lengths = [len(nodes) for nodes in levels.values()]
        # print(
        #     f"Deciding {len(playable_actions)} => {level_lengths} = {sum(level_lengths)} took {time.time() - start}"
        # )
        # breakpoint()
        return best_action


ALPHABETA_DEPTH = 3


class AlphaBetaPlayer(Player):
    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        start = time.time()
        result = alphabeta(
            game.copy(),
            ALPHABETA_DEPTH,
            float("-inf"),
            float("inf"),
            self.color,
            playable_actions=playable_actions,
        )
        print("Decision Results:", len(playable_actions), time.time() - start)
        return result[0]


def alphabeta(game, depth, alpha, beta, p0_color, playable_actions=None):
    """AlphaBeta MiniMax Algorithm.

    NOTE: Sometimes returns a value, sometimes an (action, value). This is
    because some levels are state=>action, some are action=>state and in
    action=>state would probably need (action, proba, value) as return type.
    """
    if depth == 0 or game.winning_color() is not None:
        return value_fn(game, p0_color)

    maximizingPlayer = game.current_player().color == p0_color
    children = expand_spectrum(game, playable_actions=playable_actions)
    if maximizingPlayer:
        best_action = None
        best_value = float("-inf")
        for action, outprobas in children.items():
            expected_value = 0
            for (out, proba) in outprobas:
                result = alphabeta(out, depth - 1, alpha, beta, p0_color)
                value = result if depth == 1 else result[1]
                expected_value += proba * value

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                print("beta cutoff")
                break  # beta cutoff
        return best_action, best_value
    else:
        best_action = None
        best_value = float("inf")
        for action, outprobas in children.items():
            expected_value = 0
            for (out, proba) in outprobas:
                result = alphabeta(out, depth - 1, alpha, beta, p0_color)
                value = result if depth == 1 else result[1]
                expected_value += proba * value

            if expected_value < best_value:
                best_action = action
                best_value = expected_value
            beta = min(beta, best_value)
            if beta <= alpha:
                print("alpha cutoff")
                break  # beta cutoff
        return best_action, best_value


def expand_spectrum(game, playable_actions=None):
    children = defaultdict(list)

    if playable_actions is None:
        player, action_prompt = game.pop_from_queue()
        actions = game.playable_actions(player, action_prompt)
    else:
        actions = playable_actions

    for action in actions:
        outprobas = game.execute_spectrum(action)
        children[action] = outprobas
    return children  # action => (game, proba)[]
