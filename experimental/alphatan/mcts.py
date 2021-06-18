import logging
import math

import tensorflow as tf
import numpy as np

from experimental.machine_learning.players.reinforcement import ACTION_SPACE_SIZE
from experimental.dqn_player import to_action_space, from_action_space
from experimental.machine_learning.features import create_sample_vector

EPS = 1e-8
CPUCT = 1

log = logging.getLogger(__name__)


class AlphaMCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, model):
        self.model = tf.function(model)

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def search(self, game):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        current_color = game.state.current_player().color
        sample = create_sample_vector(game, current_color)
        s = tuple(sample)  # hashable s

        if s not in self.Es:
            self.Es[s] = game_end_value(game, current_color)
        if self.Es[s] != 0:  # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            # result = self.model.predict([sample])
            tensor_sample = tf.convert_to_tensor([sample])
            result = self.model(tensor_sample)
            self.Ps[s], v = (result[0][0], result[1][0])

            # TODO: Is valids correctly formulated?
            action_ints = list(map(to_action_space, game.state.playable_actions))
            valids = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
            valids[action_ints] = 1

            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(ACTION_SPACE_SIZE):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + CPUCT * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # breakpoint()
        try:
            best_action = from_action_space(a, game.state.playable_actions)
        except Exception as e:
            breakpoint()

        # Tick and give:
        game_copy = game.copy()
        game_copy.execute(best_action)

        v = self.search(game_copy)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


def game_end_value(game, current_color):
    # r: 0 if game has not ended. 1 if player won, -1 if player lost,
    #    small non-zero value for draw.
    winning_color = game.winning_color()
    value = 0
    if winning_color == current_color:
        value = 1
    elif winning_color is not None:
        value = -1
    return value
