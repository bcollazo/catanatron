import time
from pathlib import Path
import random

import numpy as np
import tensorflow as tf

# import kerastuner as kt

from catanatron_gym.features import get_feature_ordering

# from catanatron_experimental.machine_learning.players.reinforcement import (
#     FEATURES,
#     FEATURE_INDICES,
# )
def allow_feature(feature_name):
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

# ===== Configuration
DATA_DIRECTORY = "data/simple-return-1m"
DATA_SIZE = 101479  # use zcat data/mcts-playouts/labels.csv.gzip | wc
DATA_SIZE = 101479  # use zcat data/mcts-playouts/labels.csv.gzip | wc
EPOCHS = 100
BATCH_SIZE = 1024
STEPS_PER_EPOCH = DATA_SIZE // BATCH_SIZE
PREFETCH_BUFFER_SIZE = 10
LABEL_FILE = "rewards.csv.gzip"
LABEL_COLUMN = "RETURN"
VALIDATION_DATA_SIZE = 99573
VALIDATION_DATA_SIZE = 99573
VALIDATION_DATA_DIRECTORY = "data/simple-return"
VALIDATION_STEPS = VALIDATION_DATA_SIZE // BATCH_SIZE
NORMALIZATION = False
NORMALIZATION_DIRECTORY = "data/reachability"
NORMALIZATION_MEAN_PATH = Path(NORMALIZATION_DIRECTORY, "samples-mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(NORMALIZATION_DIRECTORY, "samples-variance.npy")
SHUFFLE = True
SHUFFLE_SEED = random.randint(0, 20000)
VALIDATION_SHUFFLE_SEED = random.randint(0, 20000)
SHUFFLE_BUFFER_SIZE = 100000

MODEL_NAME = "1v1-rep-a"
MODEL_PATH = f"data/models/{MODEL_NAME}"
LOG_DIR = f"data/logs/{MODEL_NAME}/{int(time.time())}"


# ===== Create Dataset Objects
# === Download Idempotently
SAMPLES_PATH = Path(DATA_DIRECTORY, "samples.csv.gzip")
LABELS_PATH = Path(DATA_DIRECTORY, LABEL_FILE)
VALIDATION_SAMPLES_PATH = Path(VALIDATION_DATA_DIRECTORY, "samples.csv.gzip")
VALIDATION_LABELS_PATH = Path(VALIDATION_DATA_DIRECTORY, LABEL_FILE)


# === Define Generator
def preprocess(samples_batch, rewards_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_samples(samples_batch)
    label = rewards_batch[LABEL_COLUMN]
    return (input1, label)


def preprocess_samples(samples_batch):
    return tf.stack([samples_batch[feature] for feature in FEATURES], axis=1)


def build_generator(dataset):
    def generator():
        for input1, label in dataset:
            yield input1, label

    return generator


print("Reading and building train dataset...")
samples = tf.data.experimental.make_csv_dataset(
    str(SAMPLES_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=FEATURES,
)
rewards = tf.data.experimental.make_csv_dataset(
    str(LABELS_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=[LABEL_COLUMN],
    column_defaults=[tf.float64],
)
train_dataset = tf.data.Dataset.zip((samples, rewards)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)

print("Reading and building test dataset...")
samples = tf.data.experimental.make_csv_dataset(
    str(VALIDATION_SAMPLES_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=VALIDATION_SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=FEATURES,
)
rewards = tf.data.experimental.make_csv_dataset(
    str(VALIDATION_LABELS_PATH),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=VALIDATION_SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=[LABEL_COLUMN],
    column_defaults=[tf.float64],
)
test_dataset = tf.data.Dataset.zip((samples, rewards)).map(preprocess)
test_dataset = tf.data.Dataset.from_generator(
    build_generator(test_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)

# ===== REGULAR MODEL
init = tf.keras.initializers.HeUniform()
inputs = tf.keras.Input(shape=(NUM_FEATURES,))
outputs = inputs

if NORMALIZATION:
    mean = np.load(NORMALIZATION_MEAN_PATH)[FEATURE_INDICES]
    variance = np.load(NORMALIZATION_VARIANCE_PATH)[FEATURE_INDICES]
    normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization(
        mean=mean, variance=variance
    )
    outputs = normalizer_layer(outputs)

# outputs = tf.keras.layers.BatchNormalization()(outputs)
# outputs = tf.keras.layers.Dense(352, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(320, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(160, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(512, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(352, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(64, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(32, activation="relu")(outputs)
outputs = tf.keras.layers.Dense(
    8, activation="relu", kernel_initializer="random_normal"
)(outputs)
outputs = tf.keras.layers.Dense(
    units=1,
    activation="sigmoid",
    kernel_initializer="random_normal",
    kernel_regularizer="l2",
)(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    metrics=["mae", "accuracy"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6, clipnorm=1),
    # optimizer="adam",
    loss="binary_crossentropy",
)
model.summary()

# ===== Keras-Tuners NN
# def model_builder(hp):
#     inputs = tf.keras.Input(shape=(NUM_FEATURES))
#     outputs = inputs
#     outputs = tf.keras.layers.Flatten()(outputs)
#     for i in range(hp.Int("num_layers", 2, 10)):
#         outputs = tf.keras.layers.Dense(
#             units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
#             activation="relu",
#         )(outputs)
#     outputs = tf.keras.layers.Dense(units=1)(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
#     model.compile(
#         metrics=["mae"],
#         optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#         loss="mse",
#     )
#     return model


# tuner = kt.Hyperband(
#     model_builder,
#     objective="val_mae",
#     max_epochs=EPOCHS,
#     factor=3,
#     directory="keras-tuner-models",
#     project_name="rep-a-value-model",
# )
# tuner.search(
#     train_dataset,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     epochs=EPOCHS,
#     validation_data=test_dataset,
#     validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=2, min_delta=1e-6),
#         tf.keras.callbacks.TensorBoard(
#             log_dir=LOG_DIR, histogram_freq=1, write_graph=True
#         ),
#     ],
# )
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"The hyperparameter search is complete.", best_hps)
# tuner.results_summary()
# model = tuner.hypermodel.build(best_hps)

# ===== Fit Final Model
start = time.time()
model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE // BATCH_SIZE,
    callbacks=[
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_mae", patience=1, min_delta=0.0001
        # ),
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True
        ),
    ],
)
print("Training took", time.time() - start)

model.save(MODEL_PATH)
print("Saved model at:", MODEL_PATH)
