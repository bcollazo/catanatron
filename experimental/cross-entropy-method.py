import time
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import kerastuner as kt

from experimental.machine_learning.players.reinforcement import ACTION_SPACE_SIZE
from experimental.train import NUM_FEATURES


# ===== Configuration
BATCH_SIZE = 32
EPOCHS = 10
PREFETCH_BUFFER_SIZE = None
LABEL_COLUMN = "DISCOUNTED_RETURN"
DATA_SIZE = 800 * 100  # estimate: 800 samples per game.
DATA_DIRECTORY = "data/random-games-separate-edge-nodes-simple-problem-big"
STEPS_PER_EPOCH = DATA_SIZE / BATCH_SIZE
VALIDATION_DATA_SIZE = 800 * 10
VALIDATION_DATA_DIRECTORY = "data/random-games-separate-edge-nodes-simple-problem"
NORMALIZATION_BATCHES = DATA_SIZE // BATCH_SIZE
NORMALIZATION_MEAN_PATH = Path(DATA_DIRECTORY, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_DIRECTORY, "variance.npy")
NORMALIZATION_OVERWRITE = False
SHUFFLE = True
SHUFFLE_SEED = 123
INPUT_SHAPE = (NUM_FEATURES,)
OUTPUT_SHAPE = ACTION_SPACE_SIZE

# ===== Data Pipeline
def preprocess(samples_batch, actions_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_samples(samples_batch)
    label = preprocess_actions(actions_batch)
    return (input1, label)


def preprocess_samples(samples_batch):
    return tf.stack([tensor for _, tensor in samples_batch.items()], axis=1)


def preprocess_actions(actions_batch):
    return tf.stack([tensor for _, tensor in actions_batch.items()], axis=1)


def build_generator(dataset):
    def generator():
        for input1, label in dataset:
            # breakpoint()
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
)
actions = tf.data.experimental.make_csv_dataset(
    os.path.join(DATA_DIRECTORY, "actions.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
)
train_dataset = tf.data.Dataset.zip((samples, actions)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None, ACTION_SPACE_SIZE), dtype=tf.float32),
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
)
actions = tf.data.experimental.make_csv_dataset(
    os.path.join(VALIDATION_DATA_DIRECTORY, "actions.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
)
test_dataset = tf.data.Dataset.zip((samples, actions)).map(preprocess)
test_dataset = tf.data.Dataset.from_generator(
    build_generator(test_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None, ACTION_SPACE_SIZE), dtype=tf.float32),
    ),
)

# ===== Normalize features
if NORMALIZATION_OVERWRITE or (
    not NORMALIZATION_MEAN_PATH.is_file() or not NORMALIZATION_VARIANCE_PATH.is_file()
):
    print(
        f"Creating normalization layers with {NORMALIZATION_BATCHES} batches of data..."
    )
    t1 = time.time()
    normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer_layer.adapt(samples.take(NORMALIZATION_BATCHES).map(preprocess_samples))
    print(normalizer_layer.mean, normalizer_layer.variance)
    print("Took", time.time() - t1)
    np.save(NORMALIZATION_MEAN_PATH, normalizer_layer.mean)
    np.save(NORMALIZATION_VARIANCE_PATH, normalizer_layer.variance)
else:
    print(f"Reading normalization layers from disk...")
    mean = np.load(NORMALIZATION_MEAN_PATH)
    variance = np.load(NORMALIZATION_VARIANCE_PATH)
    normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization(
        mean=mean, variance=variance
    )

# ===== Select Samples to use.
# TODO: Get the rewards statistics so that we can pick the best K games.
# Iterate over
# TODO: Get Action Statistics of these top K games,
#   so that we can balance out the "End Turn" imbalance.

# ===== Build Model
def model_builder(hp):
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    outputs = inputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)
    for i in range(hp.Int("num_layers", 2, 10)):
        outputs = tf.keras.layers.Dense(
            units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
            activation="relu",
        )(outputs)

    outputs = tf.keras.layers.Dense(units=OUTPUT_SHAPE)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            "mean_absolute_error",
        ],
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate, clipnorm=1),
        loss="mean_squared_error",
    )
    return model


tuner = kt.Hyperband(
    model_builder,
    objective="val_loss",
    max_epochs=10,
    factor=3,
    directory="cross_entropy_dir",
    project_name="catanatron_project",
)
tuner.search(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
)
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The hyperparameter search is complete.", best_hps)
# Show a summary of the search
tuner.results_summary()

breakpoint()

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.summary()
model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
)

breakpoint()
timestr = time.strftime("%Y%m%d-%H%M%S")
model_path = Path("models", f"{timestr}-cross-entropy-model")
dot_img_path = model_path.with_suffix(".png")
tf.keras.utils.plot_model(model, to_file=str(dot_img_path), show_shapes=True)
model.save(str(model_path))
