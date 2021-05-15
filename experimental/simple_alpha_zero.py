from collections import deque
from experimental.dqn_player import epsilon_greedy_policy, from_action_space
from experimental.machine_learning.features import (
    create_sample_vector,
    get_feature_ordering,
)
import random

import tensorflow as tf

from catanatron.models.player import Color, Player, RandomPlayer
from experimental.play import play_batch
from experimental.machine_learning.players.reinforcement import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    normalize_action,
)

FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)

REPLAY_MEMORY_SIZE = 10_000
NUM_ITERATIONS = 10
NUM_EPISODES_PER_ITERATION = 3
NUM_GAMES_TO_PIT = 10
MODEL_ACCEPTANCE_TRESHOLD = 0.55


def main():
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    # Ensure pit initially is equal to random play.
    # Ensure doing 2 alpha zeros against each other is still random (same number of turns in avg...)
    players = [AlphaTan(Color.ORANGE, "FOO"), RandomPlayer(Color.WHITE, "BAR")]
    wins, vp_history = play_batch(100, players, None, False, False)
    breakpoint()

    # initialize neural network
    model = create_model()

    # for each iteration.
    # for each episode in iteration
    #   play game and produce many (s_t, policy_t, _) samples.
    for i in range(NUM_ITERATIONS):
        for e in range(NUM_EPISODES_PER_ITERATION):
            # TODO: self-play
            pass

        # create new neural network from old one
        candidate_model = create_model()
        candidate_model.set_weights(model.get_weights())

        # train new neural network
        # TODO:

        # pit new vs old (with temp=0). If new wins, replace.
        players = [
            AlphaTan(Color.ORANGE, "FOO", model),
            AlphaTan(Color.WHITE, "BAR", candidate_model),
        ]
        wins, vp_history = play_batch(NUM_GAMES_TO_PIT, players, None, False, False)
        if wins[Color.WHITE] >= int(sum(wins.values()) * MODEL_ACCEPTANCE_TRESHOLD):
            print("Accepting model")
            model = candidate_model
        else:
            print("Rejected model")


class AlphaTan(Player):
    def __init__(self, color, name, model=None):
        super().__init__(color, name=name)
        self.model = model or create_model()

    def decide(self, game, playable_actions):
        sample = create_sample_vector(game, self.color)
        policy = self.model.call(tf.convert_to_tensor([sample]))

        best_action_int = epsilon_greedy_policy(playable_actions, policy, 0.0)
        best_action = from_action_space(best_action_int, playable_actions)
        return best_action


def create_model():
    inputs = tf.keras.Input(shape=(NUM_FEATURES,))
    outputs = tf.keras.layers.Dense(32, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(ACTION_SPACE_SIZE)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
    return model


if __name__ == "__main__":
    main()
