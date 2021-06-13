import random
import time
from collections import deque
import uuid

import numpy as np
from tensorflow.python.keras import regularizers
from tqdm import tqdm
import tensorflow as tf


from catanatron.game import Game
from experimental.alphatan.mcts import AlphaMCTS, game_end_value
from catanatron.models.player import Color, Player, RandomPlayer
from experimental.play import play_batch
from experimental.machine_learning.players.reinforcement import ACTION_SPACE_SIZE
from experimental.dqn_player import (
    epsilon_greedy_policy,
    from_action_space,
    to_action_space,
)
from experimental.machine_learning.features import (
    create_sample_vector,
    get_feature_ordering,
)

FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)

REPLAY_MEMORY_SIZE = 10_000
TURNS_LIMIT = 10
NUM_ITERATIONS = 10
NUM_EPISODES_PER_ITERATION = 3
NUM_GAMES_TO_PIT = 10
MODEL_ACCEPTANCE_TRESHOLD = 0.55

NUM_SIMULATIONS_PER_TURN = 5
TEMP_TRESHOLD = 30

EPOCHS = 10
BATCH_SIZE = 32


def main():
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    # initialize neural network
    model = create_model()

    # # TODO: Ensure AlphaTan plays better than random.
    # players = [
    #     AlphaTan(Color.ORANGE, "FOO", model, tmp=1),
    #     RandomPlayer(Color.WHITE, "BAR"),
    # ]
    # wins, vp_history = play_batch(100, players, None, False, False, True)
    # breakpoint()

    # for each iteration.
    # for each episode in iteration
    #   play game and produce many (s_t, policy_t, _) samples.
    for i in tqdm(range(NUM_ITERATIONS), unit="iteration"):
        for e in tqdm(range(NUM_EPISODES_PER_ITERATION), unit="episode"):
            replay_memory += execute_episode(model)

        # create new neural network from old one
        candidate_model = create_model()
        candidate_model.set_weights(model.get_weights())

        # train new neural network
        random.shuffle(replay_memory)
        train(candidate_model, replay_memory)

        # pit new vs old (with temp=0). If new wins, replace.
        players = [
            AlphaTan(Color.ORANGE, model, temp=0),
            AlphaTan(Color.WHITE, candidate_model, temp=0),
        ]
        wins, vp_history = play_batch(
            NUM_GAMES_TO_PIT, players, None, False, False, False
        )
        if wins[Color.WHITE] >= int(sum(wins.values()) * MODEL_ACCEPTANCE_TRESHOLD):
            print("Accepting model")
            model = candidate_model
        else:
            print("Rejected model")


def execute_episode(model):
    mcts = AlphaMCTS(model)  # so that its shared
    p0 = AlphaTan(Color.ORANGE, model, mcts)
    p1 = AlphaTan(Color.WHITE, model, mcts)
    game = Game(players=[p0, p1])
    game.play()

    winning_player = game.winning_player()
    winning_player = game.state.players[0]
    if winning_player is None:
        return []

    winning_color = winning_player.color
    examples = [
        (state, pi, 1 if color == winning_color else -1)
        for (color, state, pi) in p0.logs + p1.logs
    ]
    return examples


def train(model, replay_memory):
    input_boards, target_pis, target_vs = list(zip(*replay_memory))
    input_boards = np.asarray(input_boards)
    target_pis = np.asarray(target_pis)
    target_vs = np.asarray(target_vs)
    model.fit(
        x=input_boards, y=[target_pis, target_vs], batch_size=BATCH_SIZE, epochs=EPOCHS
    )


class AlphaTan(Player):
    def __init__(self, color, model, mcts, temp=None):
        super().__init__(color)
        self.model = model
        self.mcts = mcts or AlphaMCTS(model)
        self.temp = temp
        self.logs = []

    def reset_state(self):
        self.mcts = AlphaMCTS(self.model)
        self.logs = []

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        start = time.time()

        temp = self.temp or int(game.state.num_turns < TEMP_TRESHOLD)
        for _ in range(NUM_SIMULATIONS_PER_TURN):
            self.mcts.search(game)

        sample = create_sample_vector(game, self.color)
        s = tuple(sample)  # hashable s
        counts = [
            self.mcts.Nsa[(s, a)] if (s, a) in self.mcts.Nsa else 0
            for a in range(ACTION_SPACE_SIZE)
        ]

        # TODO: I think playable_actions is not needed b.c. search and counts force
        #   not-playable actions to have a 0.
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            pi = [0] * len(counts)
            pi[bestA] = 1
        else:
            counts = [x ** (1.0 / temp) for x in counts]
            counts_sum = float(sum(counts))
            pi = [x / counts_sum for x in counts]

        action = np.random.choice(len(pi), p=pi)
        best_action = from_action_space(action, game.state.playable_actions)
        self.logs.append((self.color, sample, pi))

        # print("AlphaTan decision took", time.time() - start)
        return best_action


def create_model():
    inputs = tf.keras.Input(shape=(NUM_FEATURES,))

    outputs = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer="l2")(
        inputs
    )

    pi_output = tf.keras.layers.Dense(
        ACTION_SPACE_SIZE, activation="softmax", kernel_regularizer="l2"
    )(outputs)
    v_output = tf.keras.layers.Dense(1, activation="tanh")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=[pi_output, v_output])

    model.compile(
        loss=["categorical_crossentropy", "mse"],
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    main()
