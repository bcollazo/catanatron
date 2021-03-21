import time
import os
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
import kerastuner as kt

from experimental.machine_learning.players.reinforcement import FEATURES

# ===== Configuration
BATCH_SIZE = 32
EPOCHS = 10
PREFETCH_BUFFER_SIZE = 10
DATA_SIZE = 800 * 1000  # use zcat data/mcts-playouts/labels.csv.gzip | wc
DATA_DIRECTORY = "data/random-1v1s"
STEPS_PER_EPOCH = DATA_SIZE // BATCH_SIZE
LABEL_FILE = "rewards.csv.gzip"
LABEL_COLUMN = "VICTORY_POINTS_RETURN"
VALIDATION_DATA_SIZE = 800 * 1000
VALIDATION_DATA_DIRECTORY = "data/random-1v1s"
VALIDATION_STEPS = VALIDATION_DATA_SIZE // BATCH_SIZE
NORMALIZATION = False
NORMALIZATION_DIRECTORY = "data/random-1v1s"
NORMALIZATION_MEAN_PATH = Path(NORMALIZATION_DIRECTORY, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(NORMALIZATION_DIRECTORY, "variance.npy")
SHUFFLE = True
SHUFFLE_SEED = random.randint(0, 20000)
SHUFFLE_BUFFER_SIZE = 1000

NUM_FEATURES = len(FEATURES)
MODEL_NAME = "mcts-rep-a"
MODEL_PATH = f"experimental/models/{MODEL_NAME}"
LOG_DIR = f"logs/rep-a-value-model/{MODEL_NAME}/{int(time.time())}"


# ===== Building Dataset Generator
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
    os.path.join(DATA_DIRECTORY, "samples.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=FEATURES,
)
rewards = tf.data.experimental.make_csv_dataset(
    os.path.join(DATA_DIRECTORY, "labels.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=[LABEL_COLUMN],
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
    os.path.join(VALIDATION_DATA_DIRECTORY, "samples.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=FEATURES,
)
rewards = tf.data.experimental.make_csv_dataset(
    os.path.join(VALIDATION_DATA_DIRECTORY, "labels.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=[LABEL_COLUMN],
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
inputs = tf.keras.Input(shape=(NUM_FEATURES,))
outputs = inputs

# mean = np.load(NORMALIZATION_MEAN_PATH)
# variance = np.load(NORMALIZATION_VARIANCE_PATH)
# normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization(
#     mean=mean, variance=variance
# )

# outputs = normalizer_layer(outputs)
# outputs = tf.keras.layers.BatchNormalization()(outputs)
outputs = tf.keras.layers.Dense(352, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(320, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(160, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(512, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(352, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(64, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(8, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(units=1)(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    metrics=["mae"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
    loss="mean_squared_error",
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
model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
    callbacks=[
        # tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, min_delta=1e-6),
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True
        ),
    ],
)


model.save(MODEL_PATH)
print("Saved model at:", MODEL_PATH)
