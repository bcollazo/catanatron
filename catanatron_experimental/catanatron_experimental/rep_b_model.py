import time
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import kerastuner as kt

from catanatron_gym.features import get_feature_ordering
from catanatron_experimental.machine_learning.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    NUMERIC_FEATURES,
    NUM_NUMERIC_FEATURES,
)
from catanatron_experimental.datasets import (
    preprocess_samples,
    preprocess_board_tensors,
)

# ===== Configuration
BATCH_SIZE = 32
EPOCHS = 1
PREFETCH_BUFFER_SIZE = 10
LABEL_FILENAME = "rewards.csv.gzip"
LABEL_COLUMN = "prod_vps"
DATA_SIZE = 304833  # use zcat data/mcts-playouts/labels.csv.gzip | wc
DATA_SIZE = 1000 * 800  # use zcat data/mcts-playouts/labels.csv.gzip | wc
DATA_DIRECTORY = "data/random-games"
STEPS_PER_EPOCH = DATA_SIZE // BATCH_SIZE
VALIDATION_DATA_SIZE = 1000
VALIDATION_DATA_DIRECTORY = "data/random-games"
VALIDATION_STEPS = VALIDATION_DATA_SIZE // BATCH_SIZE
SHUFFLE = True
SHUFFLE_SEED = random.randint(0, 20000)
# SHUFFLE_SEED = 1
SHUFFLE_BUFFER_SIZE = 1000
STRIDES = (2, 2, 1)
USE_NUMERIC_FEATURES = False

NORMALIZE_LAYERS = False
NORMALIZATION_DATA_DIR = "data/random-games"
NUMERIC_FEATURES_MEAN_PATH = Path(NORMALIZATION_DATA_DIR, "numeric-features-mean.npy")
NUMERIC_FEATURES_VARIANCE_PATH = Path(
    NORMALIZATION_DATA_DIR, "numeric-features-variance.npy"
)
BOARD_TENSORS_MEAN_PATH = Path(NORMALIZATION_DATA_DIR, "board-tensors-mean.npy")
BOARD_TENSORS_VARIANCE_PATH = Path(NORMALIZATION_DATA_DIR, "board-tensors-variance.npy")

MODEL_NAME = "mcts-rep-b"
MODEL_PATH = f"data/models/{MODEL_NAME}"
LOG_DIR = f"logs/rep-b-mixed-data-model/{MODEL_NAME}/{int(time.time())}"


# ===== Building Dataset Generator
def preprocess(board_tensors_batch, samples_batch, rewards_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_board_tensors(board_tensors_batch, BATCH_SIZE)
    input2 = preprocess_samples(samples_batch, NUMERIC_FEATURES)
    label = rewards_batch[LABEL_COLUMN]
    return (input1, input2, label)


def build_generator(dataset):
    def generator():
        for input1, input2, label in dataset:
            yield (input1, input2), label

    return generator


def main():
    print("Reading and building train dataset...")
    board_tensors = tf.data.experimental.make_csv_dataset(
        os.path.join(DATA_DIRECTORY, "board_tensors.csv.gzip"),
        BATCH_SIZE,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
        compression_type="GZIP",
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    samples = tf.data.experimental.make_csv_dataset(
        os.path.join(DATA_DIRECTORY, "samples.csv.gzip"),
        BATCH_SIZE,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
        compression_type="GZIP",
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    rewards = tf.data.experimental.make_csv_dataset(
        os.path.join(DATA_DIRECTORY, LABEL_FILENAME),
        BATCH_SIZE,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
        compression_type="GZIP",
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        select_columns=[LABEL_COLUMN],
    )
    train_dataset = tf.data.Dataset.zip((board_tensors, samples, rewards)).map(
        preprocess
    )
    train_dataset = tf.data.Dataset.from_generator(
        build_generator(train_dataset),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(None, WIDTH, HEIGHT, CHANNELS, 1), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None, NUM_NUMERIC_FEATURES), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    print("Reading and building test dataset...")
    board_tensors = tf.data.experimental.make_csv_dataset(
        os.path.join(VALIDATION_DATA_DIRECTORY, "board_tensors.csv.gzip"),
        BATCH_SIZE,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
        compression_type="GZIP",
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    samples = tf.data.experimental.make_csv_dataset(
        os.path.join(VALIDATION_DATA_DIRECTORY, "samples.csv.gzip"),
        BATCH_SIZE,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
        compression_type="GZIP",
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    rewards = tf.data.experimental.make_csv_dataset(
        os.path.join(VALIDATION_DATA_DIRECTORY, LABEL_FILENAME),
        BATCH_SIZE,
        prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
        compression_type="GZIP",
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        select_columns=[LABEL_COLUMN],
    )
    test_dataset = tf.data.Dataset.zip((board_tensors, samples, rewards)).map(
        preprocess
    )
    test_dataset = tf.data.Dataset.from_generator(
        build_generator(test_dataset),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(None, WIDTH, HEIGHT, CHANNELS, 1), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None, NUM_NUMERIC_FEATURES), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    model = build_model()

    # ===== Fit Final Model
    model.fit(
        train_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=test_dataset,
        validation_steps=VALIDATION_STEPS,
        callbacks=[
            # tf.keras.callbacks.EarlyStopping(
            #     monitor="val_mae", patience=10, min_delta=1e-6
            # ),
            tf.keras.callbacks.TensorBoard(
                log_dir=LOG_DIR, histogram_freq=1, write_graph=True
            ),
        ],
    )

    model.save(MODEL_PATH)
    print("Saved model at:", MODEL_PATH)


# ===== REGULAR MODEL
def build_model():
    input_1 = tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS, 1))
    x = input_1

    # Normalize Input. Even tho previous games will probably miss many board
    #   configurations, we hope CNN translation invariance will compensate
    if NORMALIZE_LAYERS:
        mean = np.load(BOARD_TENSORS_MEAN_PATH)
        variance = np.load(BOARD_TENSORS_VARIANCE_PATH)
        normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=mean, variance=variance
        )
        x = normalizer_layer(x)

    # NOTE: I think last stride doestn matter
    filters_1 = 1
    DROPOUT_RATE = 0.2
    x = tf.keras.layers.Conv3D(
        filters_1,
        kernel_size=(5, 3, CHANNELS),
        strides=STRIDES,
        data_format="channels_last",
        # activation="relu",
        # kernel_constraint=tf.keras.constraints.unit_norm(),
        # kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    # reshape so that we have filters_1 "channels" of the board, and we can do Conv2D
    x = tf.keras.layers.Reshape((9, 5, filters_1))(x)  # now each pixel is a tile
    # take tile triples hierarchy.
    x = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        data_format="channels_last",
        # activation="relu",
        # kernel_constraint=tf.keras.constraints.unit_norm(),
        # kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.Model(inputs=input_1, outputs=x)

    input_2 = tf.keras.layers.Input(shape=(NUM_NUMERIC_FEATURES,))
    y = input_2
    if NORMALIZE_LAYERS:
        mean = np.load(NUMERIC_FEATURES_MEAN_PATH)
        variance = np.load(NUMERIC_FEATURES_VARIANCE_PATH)
        normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=mean, variance=variance
        )
        y = normalizer_layer(y)
    y = tf.keras.Model(inputs=input_2, outputs=y)

    combined = x.output
    if USE_NUMERIC_FEATURES:
        combined = tf.keras.layers.concatenate([x.output, y.output])

    z = tf.keras.layers.Dense(8, activation="relu")(combined)
    z = tf.keras.layers.Dense(1, activation="linear")(z)
    model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)
    model.compile(
        metrics=["mae"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
    )
    model.summary()
    return model


if __name__ == "__main__":
    main()
