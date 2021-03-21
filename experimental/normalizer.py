"""
This script will read data from a directory, and 
will save statistics to disk, so that a Normalizer
can be built

mean = np.load(NORMALIZATION_MEAN_PATH)
variance = np.load(NORMALIZATION_VARIANCE_PATH)
normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization(
    mean=mean, variance=variance
)
"""
import sys
import time
from pathlib import Path

import tensorflow as tf
import numpy as np

from experimental.machine_learning.board_tensor_features import (
    get_numeric_features,
    get_channels,
)
from experimental.datasets import (
    read_dataset,
    preprocess_samples,
    preprocess_board_tensors,
)

DATA_DIRECTORY = sys.argv[1]
NORMALIZATION_GAMES = int(sys.argv[2])
NUM_PLAYERS = int(sys.argv[3])
NORMALIZATION_BATCHES = (NORMALIZATION_GAMES * 800) // 32
SAMPLES_MEAN_PATH = Path(DATA_DIRECTORY, "samples-mean.npy")
SAMPLES_VARIANCE_PATH = Path(DATA_DIRECTORY, "samples-variance.npy")
NUMERIC_FEATURES_MEAN_PATH = Path(DATA_DIRECTORY, "numeric-features-mean.npy")
NUMERIC_FEATURES_VARIANCE_PATH = Path(DATA_DIRECTORY, "numeric-features-variance.npy")
BOARD_TENSORS_MEAN_PATH = Path(DATA_DIRECTORY, "board-tensors-mean.npy")
BOARD_TENSORS_VARIANCE_PATH = Path(DATA_DIRECTORY, "board-tensors-variance.npy")

print("Reading and building train dataset...")
samples = read_dataset(str(Path(DATA_DIRECTORY, "samples.csv.gzip")), shuffle=False)
board_tensors = read_dataset(
    str(Path(DATA_DIRECTORY, "board_tensors.csv.gzip")), shuffle=False
)

# ===== SAMPLES
print(f"Using {NORMALIZATION_BATCHES} batches of data in {DATA_DIRECTORY}...")
t1 = time.time()
normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer_layer.adapt(samples.take(NORMALIZATION_BATCHES).map(preprocess_samples))
print(normalizer_layer.mean, normalizer_layer.variance)
print("Took 1", time.time() - t1)
np.save(SAMPLES_MEAN_PATH, normalizer_layer.mean)
np.save(SAMPLES_VARIANCE_PATH, normalizer_layer.variance)
print(f"Saved to {SAMPLES_MEAN_PATH} and {SAMPLES_VARIANCE_PATH}")

# ====== BOARD TENSORS
print(f"Using {NORMALIZATION_BATCHES} batches of data in {DATA_DIRECTORY}")
t1 = time.time()
normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer_layer.adapt(
    board_tensors.take(NORMALIZATION_BATCHES).map(
        lambda d: preprocess_board_tensors(d, 32, get_channels(NUM_PLAYERS))
    )
)
print(normalizer_layer.mean, normalizer_layer.variance)
print("Took 2", time.time() - t1)
np.save(BOARD_TENSORS_MEAN_PATH, normalizer_layer.mean)
np.save(BOARD_TENSORS_VARIANCE_PATH, normalizer_layer.variance)
print(f"Saved to {BOARD_TENSORS_MEAN_PATH} and {BOARD_TENSORS_VARIANCE_PATH}")


# ====== SAMPLES NUMERIC FEATURES
print(f"Using {NORMALIZATION_BATCHES} batches of data in {DATA_DIRECTORY}")
t1 = time.time()
normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer_layer.adapt(
    samples.take(NORMALIZATION_BATCHES).map(
        lambda d: preprocess_samples(d, get_numeric_features(NUM_PLAYERS))
    )
)
print(normalizer_layer.mean, normalizer_layer.variance)
print("Took 1", time.time() - t1)
np.save(NUMERIC_FEATURES_MEAN_PATH, normalizer_layer.mean)
np.save(NUMERIC_FEATURES_VARIANCE_PATH, normalizer_layer.variance)
print(f"Saved to {NUMERIC_FEATURES_MEAN_PATH} and {NUMERIC_FEATURES_VARIANCE_PATH}")
