from pathlib import Path

import tensorflow as tf


def read_data(
    data_directory,
    batch_size=32,
    prefetch_buffer_size=None,
    shuffle=True,
    shuffle_seed=True,
    num_epochs=None,
):
    print("Reading and building train dataset...")
    samples = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "samples.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    board_tensors = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "board_tensors.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    actions = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "actions.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    rewards = tf.data.experimental.make_csv_dataset(
        str(Path(data_directory, "rewards.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    return samples, board_tensors, actions, rewards
