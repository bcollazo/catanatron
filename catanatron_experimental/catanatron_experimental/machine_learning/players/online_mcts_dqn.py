from collections import deque
from catanatron_experimental.data_logger import DataLogger
import os
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
)
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.players.playouts import run_playouts
from catanatron.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron.gym.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    create_board_tensor,
)

# ===== CONFIGURATION
NUM_FEATURES = len(get_feature_ordering())
NUM_PLAYOUTS = 100
MIN_REPLAY_BUFFER_LENGTH = 100
BATCH_SIZE = 64
FLUSH_EVERY = 1  # decisions. what takes a while is to generate samples via MCTS
TRAIN = True
OVERWRITE_MODEL = False
DATA_PATH = "data/mcts-playouts-validation"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")

# ===== PLAYER STATE (here to allow pickle-serialization of player)
# MODEL_NAME = "online-mcts-dqn-3.0"
# MODEL_PATH = str(Path("data/models/", MODEL_NAME))
# MODEL_SINGLETON = None
DATA_LOGGER = DataLogger(DATA_PATH)


# def get_model():
#     global MODEL_SINGLETON
#     if MODEL_SINGLETON is None:
#         if os.path.isdir(MODEL_PATH):
#             MODEL_SINGLETON = tf.keras.models.load_model(MODEL_PATH)
#         else:
#             MODEL_SINGLETON = create_model()
#     return MODEL_SINGLETON


# def create_model():
#     inputs = Input(shape=(NUM_FEATURES,))
#     outputs = inputs

#     # mean = np.load(NORMALIZATION_MEAN_PATH)
#     # variance = np.load(NORMALIZATION_VARIANCE_PATH)
#     # normalizer_layer = Normalization(mean=mean, variance=variance)
#     # outputs = normalizer_layer(outputs)

#     # outputs = Dense(8, activation="relu")(outputs)

#     # TODO: We may want to change infra to predict all 4 winning probas.
#     #   So that mini-max makes more sense? Enemies wont min you, they'll max
#     #   themselves.
#     outputs = Dense(units=1, activation="linear")(outputs)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
#     return model


class OnlineMCTSDQNPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self.step = 0

    def decide(self, game: Game, playable_actions):
        """
        For each move, will run N playouts, get statistics, and save into replay buffer.
        Every M decisions, will:
            - flush replay buffer to disk (for offline experiments)
            - report progress on games thus far to TensorBoard (tf.summary module)
            - update model by choosing L random samples from replay buffer
                and train model. do we need stability check? i think not.
                and override model path.
        Decision V1 looks like, predict and choose the one that creates biggest
            'distance' against enemies. Actually this is the same as maximizing wins.
        Decision V2 looks the same as V1, but minimaxed some turns in the future.
        """
        if len(playable_actions) == 1:  # this avoids imbalance (if policy-learning)
            return playable_actions[0]

        start = time.time()

        # Run MCTS playouts for each possible action, save results for training.
        samples = []
        scores = []
        print(playable_actions)
        for action in playable_actions:
            print("Considering", action)
            action_applied_game_copy = game.copy()
            action_applied_game_copy.execute(action)
            sample = create_sample_vector(action_applied_game_copy, self.color)
            samples.append(sample)

            if TRAIN:
                # Save snapshots from the perspective of each player (more training!)
                counter = run_playouts(action_applied_game_copy, NUM_PLAYOUTS)
                mcts_labels = {k: v / NUM_PLAYOUTS for k, v in counter.items()}
                DATA_LOGGER.consume(action_applied_game_copy, mcts_labels)

                scores.append(mcts_labels.get(self.color, 0))

        # TODO: if M step, do all 4 things.
        if TRAIN and self.step % FLUSH_EVERY == 0:
            self.update_model_and_flush_samples()

        # scores = get_model().call(tf.convert_to_tensor(samples))
        best_idx = np.argmax(scores)
        best_action = playable_actions[best_idx]

        if TRAIN:
            print("Decision took:", time.time() - start)
        self.step += 1
        return best_action

    def update_model_and_flush_samples(self):
        """Trains using NN, and saves to disk"""
        global MIN_REPLAY_BUFFER_LENGTH, BATCH_SIZE, MODEL_PATH, OVERWRITE_MODEL

        samples, board_tensors, labels = DATA_LOGGER.get_replay_buffer()
        if len(samples) < MIN_REPLAY_BUFFER_LENGTH:
            return

        # print("Training...")
        # model = get_model()
        # model.fit(
        #     tf.convert_to_tensor(samples),
        #     tf.convert_to_tensor(labels),
        #     batch_size=BATCH_SIZE,
        #     verbose=0,
        #     shuffle=True,
        # )
        # print("DONE training")
        # if OVERWRITE_MODEL:
        #     model.save(MODEL_PATH)

        DATA_LOGGER.flush()
