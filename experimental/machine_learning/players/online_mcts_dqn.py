from collections import deque
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
from experimental.machine_learning.players.mcts import run_playouts
from experimental.machine_learning.features import (
    create_sample_vector,
    get_feature_ordering,
    iter_players,
)
from experimental.machine_learning.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    create_board_tensor,
)

# ===== CONFIGURATION
NUM_FEATURES = len(get_feature_ordering())
NUM_PLAYOUTS = 25
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_BUFFER_LENGTH = 1_000
# MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
BATCH_SIZE = 64
FLUSH_EVERY = 1  # decisions. what takes a while is to generate samples via MCTS
TRAIN = True
NORMALIZATION_DIRECTORY = "data/random-games"
NORMALIZATION_MEAN_PATH = Path(NORMALIZATION_DIRECTORY, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(NORMALIZATION_DIRECTORY, "variance.npy")
OUTPUT_MCTS_DATA_PATH = "data/mcts-playouts-labeling-2"

# ===== PLAYER STATE (here to allow pickle-serialization of player)
MODEL_NAME = "online-mcts-dqn-2"
MODEL_PATH = str(Path("experimental/models/", MODEL_NAME))
MODEL_SINGLETON = None
# for now will only be (sample, label) pairs. could be (state, action, reward, next_state)
REPLAY_BUFFER = []


def get_model():
    global MODEL_SINGLETON
    if MODEL_SINGLETON is None:
        if os.path.isdir(MODEL_PATH):
            MODEL_SINGLETON = tf.keras.models.load_model(MODEL_PATH)
        else:
            MODEL_SINGLETON = create_model()
    return MODEL_SINGLETON


def create_model():
    inputs = Input(shape=(NUM_FEATURES,))
    outputs = inputs

    mean = np.load(NORMALIZATION_MEAN_PATH)
    variance = np.load(NORMALIZATION_VARIANCE_PATH)
    normalizer_layer = Normalization(mean=mean, variance=variance)
    outputs = normalizer_layer(outputs)

    # outputs = Dense(8, activation="relu")(outputs)

    # TODO: We may want to change infra to predict all 4 winning probas.
    #   So that mini-max makes more sense? Enemies wont min you, they'll max
    #   themselves.
    outputs = Dense(units=1, activation="linear")(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
    return model


class OnlineMCTSDQNPlayer(Player):
    def __init__(self, color, name):
        super().__init__(color, name=name)
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
        global REPLAY_BUFFER
        if len(playable_actions) == 1:  # this avoids imbalance (if policy-learning)
            return playable_actions[0]

        start = time.time()

        # Run MCTS playouts for each possible action, save results for training.
        samples = []
        for action in playable_actions:
            action_applied_game_copy = game.copy()
            action_applied_game_copy.execute(action)
            sample = create_sample_vector(action_applied_game_copy, self)
            samples.append(sample)

            if TRAIN:
                # Save snapshots from the perspective of each player (more training!)
                counter = run_playouts(action_applied_game_copy, NUM_PLAYOUTS)
                for i, player in iter_players(game, self):
                    if i == 0:
                        continue
                    sample = create_sample_vector(action_applied_game_copy, player)
                    flattened_board_tensor = tf.reshape(
                        create_board_tensor(action_applied_game_copy, player),
                        (WIDTH * HEIGHT * CHANNELS,),
                    ).numpy()
                    label = counter[player.color] / NUM_PLAYOUTS
                    REPLAY_BUFFER.append((sample, flattened_board_tensor, label))

        # TODO: if M step, do all 4 things.
        if TRAIN and self.step % FLUSH_EVERY == 0:
            self.update_model_and_flush_samples()

        scores = get_model().call(tf.convert_to_tensor(samples))
        best_idx = np.argmax(scores)
        best_action = playable_actions[best_idx]

        if TRAIN:
            print("Decision took:", time.time() - start)
        self.step += 1
        return best_action

    def update_model_and_flush_samples(self):
        """Trains using NN, and saves to disk"""
        global REPLAY_BUFFER, MIN_REPLAY_BUFFER_LENGTH, BATCH_SIZE, MODEL_PATH
        if len(REPLAY_BUFFER) < MIN_REPLAY_BUFFER_LENGTH:
            return

        print("Flushing Data", len(REPLAY_BUFFER))
        X1 = []
        X2 = []
        Y = []
        for (sample, flattened_board_tensor, label) in REPLAY_BUFFER:
            X1.append(sample)
            X2.append(flattened_board_tensor)
            Y.append(label)

        breakpoint()
        print("Training...")
        model = get_model()
        model.fit(
            tf.convert_to_tensor(X1),
            tf.convert_to_tensor(Y),
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=False,
        )
        print("DONE training")
        model.save(MODEL_PATH)

        # Consume REPLAY_BUFFER
        samples_df = pd.DataFrame(X1).astype("float64")
        board_tensors_df = pd.DataFrame(X2).astype("float64")
        labels_df = pd.DataFrame(Y).astype("float64")
        samples_path = Path(OUTPUT_MCTS_DATA_PATH, "samples.csv.gzip")
        board_tensors_path = Path(OUTPUT_MCTS_DATA_PATH, "board_tensors.csv.gzip")
        labels_path = Path(OUTPUT_MCTS_DATA_PATH, "labels.csv.gzip")
        if not os.path.exists(OUTPUT_MCTS_DATA_PATH):
            os.makedirs(OUTPUT_MCTS_DATA_PATH)

        is_first_training = not os.path.isfile(samples_path)
        samples_df.to_csv(
            samples_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        board_tensors_df.to_csv(
            board_tensors_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        labels_df.to_csv(
            labels_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        REPLAY_BUFFER = []
