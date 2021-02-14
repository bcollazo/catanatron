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
import time
from pathlib import Path

import tensorflow as tf
import numpy as np

from experimental.datasets import read_data, preprocess_samples

NORMALIZATION_GAMES = 1000
NORMALIZATION_BATCHES = (NORMALIZATION_GAMES * 800) // 32
INPUT_DIRECTORY = "data/random-games"
NORMALIZATION_MEAN_PATH = Path(INPUT_DIRECTORY, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(INPUT_DIRECTORY, "variance.npy")

samples, board_tensors, actions, rewards = read_data(INPUT_DIRECTORY)

print(f"Creating normalization layers with {NORMALIZATION_BATCHES} batches of data...")
t1 = time.time()
normalizer_layer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer_layer.adapt(samples.take(NORMALIZATION_BATCHES).map(preprocess_samples))
print(normalizer_layer.mean, normalizer_layer.variance)
print("Took", time.time() - t1)
np.save(NORMALIZATION_MEAN_PATH, normalizer_layer.mean)
np.save(NORMALIZATION_VARIANCE_PATH, normalizer_layer.variance)
print(f"Saved to {NORMALIZATION_MEAN_PATH} and {NORMALIZATION_VARIANCE_PATH}")
