from catanatron_experimental.machine_learning.players.reinforcement import (
    ACTION_SPACE_SIZE,
)
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Read datasets. Try to fit model
from simple_alpha_zero import create_model, load_replay_memory, pit
from catanatron_experimental.machine_learning.board_tensor_features import (
    HEIGHT,
    WIDTH,
    create_board_tensor,
    get_channels,
)
from catanatron_gym.features import get_feature_ordering


def allow_feature(feature_name):
    return True
    return (
        feature_name != "IS_DISCARDING"
        and feature_name != "IS_MOVING_ROBBER"
        and "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN" not in feature_name
        and "VICTORY_POINT" not in feature_name
        and "2_ROAD_REACHABLE" not in feature_name
        and "0_ROAD_REACHABLE" not in feature_name
        and "1_ROAD_REACHABLE" not in feature_name
        and "HAND" not in feature_name
        and "BANK" not in feature_name
        and "P0_ACTUAL_VPS" != feature_name
        and "PLAYABLE" not in feature_name
        # and "LEFT" not in feature_name
        and "ROLLED" not in feature_name
        and "PLAYED" not in feature_name
        and "PUBLIC_VPS" not in feature_name
        and not ("EFFECTIVE" in feature_name and "P1" in feature_name)
        and not ("EFFECTIVE" in feature_name and "P0" in feature_name)
        and (feature_name[-6:] != "PLAYED" or "KNIGHT" in feature_name)
    )


ALL_FEATURES = get_feature_ordering(num_players=2)
FEATURES = list(filter(allow_feature, ALL_FEATURES))
NUM_FEATURES = len(FEATURES)
CHANNELS = get_channels(2)

DATA_DIRECTORY = "data/alphatan-memory"
DATA_SIZE = 377824
BATCH_SIZE = 1024
EPOCHS = 10
STEPS_PER_EPOCH = DATA_SIZE // BATCH_SIZE
# (states, board_tensors, pis, vs) = load_replay_memory(data_directory)
SHUFFLE = True
SHUFFLE_SEED = 123
PREFETCH_BUFFER_SIZE = 1000
SHUFFLE_BUFFER_SIZE = 1000

# === Download Idempotently
STATE_PATH = Path(DATA_DIRECTORY, "states.csv.gzip")
BOARD_TENSORS_PATH = Path(DATA_DIRECTORY, "board_tensors.csv.gzip")
PIS_PATH = Path(DATA_DIRECTORY, "pis.csv.gzip")
VS_PATH = Path(DATA_DIRECTORY, "vs.csv.gzip")

dfa = pd.read_csv("data/alphatan-memory/pis-mean.csv", index_col=0)
means = dfa.to_numpy()[:, 0]
class_weight = np.nan_to_num(1 / means, posinf=0)

# === Define Generator
def preprocess(states_batch, pis_batch, vs_batch):
    """Input are dictionary of tensors"""
    input1 = simple_stack(states_batch)
    label = (simple_stack(pis_batch), simple_stack(vs_batch))
    return (input1, label)


def simple_stack(batch):
    return tf.stack([batch[x] for x in batch.keys()], axis=1)


def build_generator(dataset):
    def generator():
        for input1, (pis, vs) in dataset:
            target_pis = pis.numpy()
            sample_weight = [class_weight[pi.argmax()] for pi in target_pis]
            yield input1, (pis, vs), sample_weight

    return generator


print("Reading and building train dataset...")
states = tf.data.experimental.make_csv_dataset(
    str(STATE_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=FEATURES,
)
pis = tf.data.experimental.make_csv_dataset(
    str(PIS_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
)
vs = tf.data.experimental.make_csv_dataset(
    str(VS_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
)
train_dataset = tf.data.Dataset.zip((states, pis, vs)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, ACTION_SPACE_SIZE), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)

# ===== Build Model
NEURONS_PER_LAYER = [64, 64, 64, 64, 32]
inputs = tf.keras.Input(shape=(NUM_FEATURES,))
outputs = inputs

for neurons in NEURONS_PER_LAYER:
    outputs = tf.keras.layers.Dense(neurons, activation="relu")(outputs)

pi_output = tf.keras.layers.Dense(
    ACTION_SPACE_SIZE,
    activation="softmax",
    # kernel_regularizer="l2",
)(outputs)
v_output = tf.keras.layers.Dense(1, activation="tanh")(outputs)

candidate_model = tf.keras.Model(inputs=inputs, outputs=[pi_output, v_output])
candidate_model.compile(
    loss=["categorical_crossentropy", "mse"],
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=["mae"],
)

# ===== Train
logdir = f"data/logs/offline-alphatan/{int(time.time())}"
candidate_model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss", patience=1, min_delta=0.0001
        # ),
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1, write_graph=True
        ),
    ],
)
breakpoint()
# TODO: Play against normal alphatan.
model = create_model()
result = pit(model, candidate_model)
