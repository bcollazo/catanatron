from pathlib import Path

import tensorflow as tf


def read_data(
    data_directory,
    batch_size=32,
    prefetch_buffer_size=None,
    shuffle=True,
    shuffle_seed=True,
):
    print("Reading and building train dataset...")
    samples = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "samples.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    actions = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "actions.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    rewards = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "rewards.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    return samples, actions, rewards
