import time
import random
from typing import Any
from collections import deque

from catanatron.game import Game
from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    freqdeck_subtract,
)
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.tree_search_utils import (
    expand_spectrum,
    list_prunned_actions,
)
from catanatron_experimental.machine_learning.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,
)


ALPHABETA_DEFAULT_DEPTH = 2
MAX_SEARCH_TIME_SECS = 20


class AlphaBetaPlayer(Player):
    """
    Player that executes an AlphaBeta Search where the value of each node
    is taken to be the expected value (using the probability of rolls, etc...)
    of its children. At leafs we simply use the heuristic function given.

    NOTE: More than 3 levels seems to take much longer, it would be
    interesting to see this with prunning.
    """

    def __init__(
        self,
        color,
        depth=ALPHABETA_DEFAULT_DEPTH,
        prunning=False,
        value_fn_builder_name=None,
        params=DEFAULT_WEIGHTS,
        epsilon=None,
    ):
        super().__init__(color)
        self.depth = int(depth)
        self.prunning = str(prunning).lower() != "false"
        self.value_fn_builder_name = (
            "contender_fn" if value_fn_builder_name == "C" else "base_fn"
        )
        self.params = params
        self.use_value_function = None
        self.epsilon = epsilon

    def value_function(self, game, p0_color):
        raise NotImplementedError

    def get_actions(self, game):
        if self.prunning:
            return list_prunned_actions(game)
        return game.state.playable_actions

    def decide(self, game: Game, playable_actions):
        actions = self.get_actions(game)
        if len(actions) == 1:
            return actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        start = time.time()
        state_id = str(len(game.state.actions))
        node = DebugStateNode(state_id, self.color)  # i think it comes from outside
        deadline = start + MAX_SEARCH_TIME_SECS
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline, node
        )
        # print("Decision Results:", self.depth, len(actions), time.time() - start)
        # if game.state.num_turns > 10:
        #     render_debug_tree(node)
        #     breakpoint()
        return result[0]

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(depth={self.depth},value_fn={self.value_fn_builder_name},prunning={self.prunning})"
        )

    def alphabeta(self, game, depth, alpha, beta, deadline, node):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """
        if depth == 0 or game.winning_color() is not None or time.time() >= deadline:
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            value = value_fn(game, self.color)

            node.expected_value = value
            return None, value

        maximizingPlayer = game.state.current_color() == self.color
        actions = self.get_actions(game)  # list of actions.
        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]

        if maximizingPlayer:
            best_action = None
            best_value = float("-inf")
            for i, (action, outcomes) in enumerate(action_outcomes.items()):
                action_node = DebugActionNode(action)

                expected_value = 0
                for j, (outcome, proba) in enumerate(outcomes):
                    out_node = DebugStateNode(
                        f"{node.label} {i} {j}", outcome.state.current_color()
                    )

                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline, out_node
                    )
                    value = result[1]
                    expected_value += proba * value

                    action_node.children.append(out_node)
                    action_node.probas.append(proba)

                action_node.expected_value = expected_value
                node.children.append(action_node)

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # beta cutoff

            node.expected_value = best_value
            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for i, (action, outcomes) in enumerate(action_outcomes.items()):
                action_node = DebugActionNode(action)

                expected_value = 0
                for j, (outcome, proba) in enumerate(outcomes):
                    out_node = DebugStateNode(
                        f"{node.label} {i} {j}", outcome.state.current_color()
                    )

                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline, out_node
                    )
                    value = result[1]
                    expected_value += proba * value

                    action_node.children.append(out_node)
                    action_node.probas.append(proba)

                action_node.expected_value = expected_value
                node.children.append(action_node)

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # alpha cutoff

            node.expected_value = best_value
            return best_action, best_value


class DebugStateNode:
    def __init__(self, label, color):
        self.label = label
        self.children = []  # DebugActionNode[]
        self.expected_value = None
        self.color = color


class DebugActionNode:
    def __init__(self, action):
        self.action = action
        self.expected_value: Any = None
        self.children = []  # DebugStateNode[]
        self.probas = []


def render_debug_tree(node):
    from graphviz import Digraph

    dot = Digraph("AlphaBetaSearch")

    agenda = [node]

    while len(agenda) != 0:
        tmp = agenda.pop()
        dot.node(
            tmp.label,
            label=f"<{tmp.label}<br /><font point-size='10'>{tmp.expected_value}</font>>",
            style="filled",
            fillcolor=tmp.color.value,
        )
        for child in tmp.children:
            action_label = (
                f"{tmp.label} - {str(child.action).replace('<', '').replace('>', '')}"
            )
            dot.node(
                action_label,
                label=f"<{action_label}<br /><font point-size='10'>{child.expected_value}</font>>",
                shape="box",
            )
            dot.edge(tmp.label, action_label)
            for action_child, proba in zip(child.children, child.probas):
                dot.node(
                    action_child.label,
                    label=f"<{action_child.label}<br /><font point-size='10'>{action_child.expected_value}</font>>",
                )
                dot.edge(action_label, action_child.label, label=str(proba))
                agenda.append(action_child)
    print(dot.render())


def generate_desired_domestic_trades(hand_freqdeck):
    """
    Will generate 1:1 trades that get hand to a buildable state.

    Does not consider what is "buildable" on board
    or if other players have that resource (i.e. dumb generation).
    """
    interesting = set()  # set of 1:1 trades that get you to build

    costs = [
        ROAD_COST_FREQDECK,
        SETTLEMENT_COST_FREQDECK,
        CITY_COST_FREQDECK,
        DEVELOPMENT_CARD_COST_FREQDECK,
    ]

    # Costs are 1-1-1-1-0, 1-1-0-0-0, 0-0-0-2-3, 0-0-1-1-1
    # Check how "close hand_freqdeck is to each, if missing one use"
    for cost in costs:
        diff = freqdeck_subtract(hand_freqdeck, cost)

        num_missing = sum(i < 0 for i in diff)
        only_missing_one_of_one_resource = diff.count(-1) == 1 and num_missing == 1
        if only_missing_one_of_one_resource:
            to_ask_index = diff.index(-1)
            to_ask = [0, 0, 0, 0, 0]
            to_ask[to_ask_index] = 1
            for i, _ in enumerate(RESOURCES):
                if diff[i] > 0:
                    to_give = [0, 0, 0, 0, 0]
                    to_give[i] = 1
                    interesting.add((*to_give, *to_ask))
        else:
            # theres either no -1s, so can buy, or there are more
            # than 1 -1, in which case would need several trades...
            pass

    return interesting


# TODO: Evaluate generation so as to
#   get ones that improve value respect to other player.


class CardCountingPlayer(AlphaBetaPlayer):
    """
    Keeps hand
    """

    def __init__(self, color, num_memory, is_bot=True):
        super().__init__(color, is_bot)
        self.memory = int(num_memory)
        self.distribution = None
        self.last_action_index = 0
        self.last_cards = deque([], self.memory)  # FIFO queue of "seen cards"
        self.hand_belief = {}  # color => freqdeck

    def decide(self, game, playable_actions):
        actions_catch_up = game.state.actions[self.last_action_index :]
        for action in actions_catch_up:
            self.consume(game, action)

        return super().decide(game, playable_actions)

    def consume(self, game, action):
        # if its a BUILD_ROAD, get cost, and subtract from that players hand
        is_initial_building_phase = self.last_action_index < 4 * len(game.state.colors)
        is_first_roll = self.last_action_index == 4 * len(game.state.colors)
        if is_initial_building_phase:
            pass  # skip and just get cards and last
        elif is_first_roll:
            pass
        else:
            pass

        print(self.last_action_index, is_initial_building_phase, is_first_roll)
        breakpoint()
        self.last_action_index += 1


class SameTurnAlphaBetaPlayer(AlphaBetaPlayer):
    """
    Same like AlphaBeta but only within turn
    """

    def alphabeta(self, game, depth, alpha, beta, deadline, node):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """
        if (
            depth == 0
            or game.state.current_color() != self.color
            or game.winning_color() is not None
            or time.time() >= deadline
        ):
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            value = value_fn(game, self.color)

            node.expected_value = value
            return None, value

        actions = self.get_actions(game)  # list of actions.
        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]

        best_action = None
        best_value = float("-inf")
        for i, (action, outcomes) in enumerate(action_outcomes.items()):
            action_node = DebugActionNode(action)

            expected_value = 0
            for j, (outcome, proba) in enumerate(outcomes):
                out_node = DebugStateNode(
                    f"{node.label} {i} {j}", outcome.state.current_color()
                )

                result = self.alphabeta(
                    outcome, depth - 1, alpha, beta, deadline, out_node
                )
                value = result[1]
                expected_value += proba * value

                action_node.children.append(out_node)
                action_node.probas.append(proba)

            action_node.expected_value = expected_value
            node.children.append(action_node)

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # beta cutoff

        node.expected_value = best_value
        return best_action, best_value
