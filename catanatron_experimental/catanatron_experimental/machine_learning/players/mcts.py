import math
import time
from collections import defaultdict

import numpy as np

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.playouts import run_playout
from catanatron_experimental.machine_learning.players.tree_search_utils import (
    execute_spectrum,
    list_prunned_actions,
)

SIMULATIONS = 10
epsilon = 1e-8
EXP_C = 2**0.5


class StateNode:
    def __init__(self, color, game, parent, prunning=False):
        self.level = 0 if parent is None else parent.level + 1
        self.color = color  # color of player carrying out MCTS
        self.parent = parent
        self.game = game  # state
        self.children = []
        self.prunning = prunning

        self.wins = 0
        self.visits = 0
        self.result = None  # set if terminal

    def run_simulation(self):
        # select
        tmp = self
        tmp.visits += 1
        while not tmp.is_leaf():
            tmp = tmp.select()
            tmp.visits += 1

        if not tmp.is_terminal():
            # expand
            tmp.expand()
            tmp = tmp.select()
            tmp.visits += 1

            # playout
            result = tmp.playout()
        else:
            result = self.game.winning_color()

        # backpropagate
        tmp.backpropagate(result == self.color)

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.game.winning_color() is not None

    def expand(self):
        children = defaultdict(list)
        playable_actions = self.game.state.playable_actions
        actions = list_prunned_actions(self.game) if self.prunning else playable_actions
        for action in actions:
            outcomes = execute_spectrum(self.game, action)
            for state, proba in outcomes:
                children[action].append(
                    (StateNode(self.color, state, self, self.prunning), proba)
                )
        self.children = children

    def select(self):
        """select a child StateNode"""
        action = self.choose_best_action()

        # Idea: Allow randomness to guide to next children too
        children = self.children[action]
        children_states = list(map(lambda c: c[0], children))
        children_probas = list(map(lambda c: c[1], children))
        return np.random.choice(children_states, 1, p=children_probas)[0]

    def choose_best_action(self):
        scores = []
        for action in self.game.state.playable_actions:
            score = self.action_children_expected_score(action)
            scores.append(score)

        idx = max(range(len(scores)), key=lambda i: scores[i])
        action = self.game.state.playable_actions[idx]
        return action

    def action_children_expected_score(self, action):
        score = 0
        for child, proba in self.children[action]:
            score += proba * (
                child.wins / (child.visits + epsilon)
                + EXP_C
                * (math.log(self.visits + epsilon) / (child.visits + epsilon)) ** 0.5
            )
        return score

    def playout(self):
        return run_playout(self.game)

    def backpropagate(self, value):
        self.wins += value

        tmp = self
        while tmp.parent is not None:
            tmp = tmp.parent

            tmp.wins += value


class MCTSPlayer(Player):
    def __init__(self, color, num_simulations=SIMULATIONS, prunning=False):
        super().__init__(color)
        self.num_simulations = int(num_simulations)
        self.prunning = bool(prunning)

    def decide(self, game: Game, playable_actions):
        # if len(game.state.actions) > 10:
        #     import sys

        #     sys.exit(1)
        actions = list_prunned_actions(game) if self.prunning else playable_actions
        if len(actions) == 1:
            return actions[0]

        start = time.time()
        root = StateNode(self.color, game.copy(), None, self.prunning)
        for _ in range(self.num_simulations):
            root.run_simulation()

        print(
            f"{str(self)} took {time.time() - start} secs to decide {len(playable_actions)}"
        )

        return root.choose_best_action()

    def __repr__(self):
        return super().__repr__() + f"({self.num_simulations}:{self.prunning})"
