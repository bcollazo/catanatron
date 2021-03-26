import math
import time
from collections import defaultdict

import numpy as np

from catanatron.game import Game
from catanatron.models.player import Player
from experimental.machine_learning.players.playouts import run_playout

SIMULATIONS = 10
epsilon = 1e-8
EXP_C = 2 ** 0.5


class StateNode:
    def __init__(self, color, state, parent, playable_actions=None):
        self.level = 0 if parent is None else parent.level + 1
        self.color = color  # color of player carrying out MCTS
        self.parent = parent
        self.state = state
        self.playable_actions = playable_actions
        self.children = []

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
            result = self.state.winning_player()

        # backpropagate
        tmp.backpropagate(result == self.color)

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state.winning_player() is not None

    def expand(self):
        # This method should only be called once per node
        actions = self.get_playable_actions()

        children = defaultdict(list)
        for action in actions:
            outcomes = self.state.execute_spectrum(action)
            for (state, proba) in outcomes:
                children[action].append((StateNode(self.color, state, self), proba))
        self.children = children

    def get_playable_actions(self):
        if self.playable_actions is None:
            player, action_prompt = self.state.pop_from_queue()
            actions = self.state.playable_actions(player, action_prompt)
            self.current_player = player
            self.playable_actions = actions
        return self.playable_actions

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
        for action in self.playable_actions:
            score = self.action_children_expected_score(action)
            scores.append(score)

        idx = max(range(len(scores)), key=lambda i: scores[i])
        action = self.playable_actions[idx]
        return action

    def action_children_expected_score(self, action):
        score = 0
        for (child, proba) in self.children[action]:
            score += proba * (
                child.wins / (child.visits + epsilon)
                + EXP_C
                * (math.log(self.visits + epsilon) / (child.visits + epsilon)) ** 0.5
            )
        return score

    def playout(self):
        return run_playout(self.state)

    def backpropagate(self, value):
        self.wins += value

        tmp = self
        while tmp.parent is not None:
            tmp = tmp.parent

            tmp.wins += value


class MCTSPlayer(Player):
    def __init__(self, color, name, num_simulations=SIMULATIONS):
        super().__init__(color, name=name)
        self.num_simulations = num_simulations

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        root = StateNode(self.color, game.copy(), None, playable_actions)
        for i in range(self.num_simulations):
            start = time.time()
            root.run_simulation()
            print("Simulation", i, "took", time.time() - start)

        return root.choose_best_action()
