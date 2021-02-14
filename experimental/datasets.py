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
    samples = read_dataset(
        str(Path(data_directory, "samples.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    board_tensors = read_dataset(
        str(Path(data_directory, "board_tensors.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    actions = read_dataset(
        str(Path(data_directory, "actions.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    rewards = read_dataset(
        str(Path(data_directory, "rewards.csv.gzip")),
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )
    return samples, board_tensors, actions, rewards


def read_dataset(
    path,
    batch_size=32,
    prefetch_buffer_size=None,
    shuffle=True,
    shuffle_seed=True,
    num_epochs=None,
):
    return tf.data.experimental.make_csv_dataset(
        path,
        batch_size,
        prefetch_buffer_size=prefetch_buffer_size,
        compression_type="GZIP",
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_epochs=num_epochs,
    )


def preprocess_samples(samples_batch):
    return tf.stack([tensor for _, tensor in samples_batch.items()], axis=1)


def preprocess_actions(actions_batch):
    return tf.stack([tensor for _, tensor in actions_batch.items()], axis=1)
