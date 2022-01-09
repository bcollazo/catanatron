import os
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanatron_gym.features import get_feature_ordering
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_math_ops import select
from tqdm import tqdm

DATA_DIRECTORY = "data/cross-entropy-attempt"
# DATA_DIRECTORY = "data/small-dataset"
BATCH_SIZE = 32
SHUFFLE_SEED = 1
LABEL = "DISCOUNTED_RETURN"

DATASET_DIRECTORY = os.path.join(DATA_DIRECTORY, "main.csv.gzip")
rows_per_bytes = 6345 / 1209734
size = os.path.getsize(DATASET_DIRECTORY)  # bytes
estimated_rows = size * rows_per_bytes

df = pd.read_csv(DATASET_DIRECTORY, compression="gzip", nrows=1)
breakpoint()

# ===== Determine threshold with which to filter dataset
def preprocess(batch):
    return tf.stack(batch[LABEL])


print(f"Preparing dataset... (estimated {estimated_rows}")
dataset = tf.data.experimental.make_csv_dataset(
    DATASET_DIRECTORY,
    BATCH_SIZE,
    compression_type="GZIP",
    shuffle=False,
    num_epochs=1,
    select_columns=[LABEL],
).map(preprocess)
print("Finished preparing dataset...")

print("Computing mean...")
mean = tf.keras.metrics.Mean()
for batch in tqdm(dataset, total=int(estimated_rows / BATCH_SIZE)):
    mean.update_state(batch)
print("Finished computing mean...")

thresh = mean.result().numpy()
print(thresh)

# ===== Read Dataset
def preprocess_features(batch):
    # TODO: change representation and return x, y.
    breakpoint()
    return tf.stack(
        [tf.cast(tensor, tf.float32) for _, tensor in batch.items()], axis=1
    )


print(f"Preparing dataset... (estimated {estimated_rows}")
dataset = tf.data.experimental.make_csv_dataset(
    DATASET_DIRECTORY,
    BATCH_SIZE,
    compression_type="GZIP",
    shuffle=False,
    num_epochs=1,
).map(preprocess)
print("Finished preparing dataset...")


# ===== Generator Dataset
# def build_generator(dataset):
#     def generator():
#         for element in dataset:
#             yield element
#     return generator
# NUM_FEATURES = len(get_feature_ordering())
# gen_dataset = tf.data.Dataset.from_generator(
#     build_generator(dataset),
#     output_signature=(
#         tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, ACTION_SPACE_SIZE), dtype=tf.float32),
#     ),
# )

# dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
# dataset = dataset.filter(lambda x: x < 3)
# list(dataset.as_numpy_iterator())

# # `tf.math.equal(x, y)` is required for equality comparison
# def filter_fn(x):
#   return tf.math.equal(x, 1)
# dataset = dataset.filter(filter_fn)
# list(dataset.as_numpy_iterator())

"""
Iterates over the dataset to compute mean and max of label.
Then iterates again filtering out everything with label 
"""


# should be called like:
# python cross-entropy-filter.py

# Stream read dataset to compute label statistic.
# Label should be discounted return.

# Take top 10% of games. Save them in

# Read model checkpoint if exists or start new one.
# Train on that only if data sounds enough.

# Save model.
