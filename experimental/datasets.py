from pathlib import Path

import tensorflow as tf
import pandas as pd

from experimental.machine_learning.board_tensor_features import CHANNELS, HEIGHT, WIDTH


def read_data(
    data_directory,
    batch_size=32,
    prefetch_buffer_size=None,
    shuffle=True,
    shuffle_seed=123,
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
    shuffle_seed=123,
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


def preprocess_samples(samples_batch, features=None):
    features = features or samples_batch.keys()
    return tf.stack([samples_batch[k] for k in features], axis=1)


def preprocess_actions(actions_batch):
    return tf.stack([tensor for _, tensor in actions_batch.items()], axis=1)


def preprocess_board_tensors(board_tensors_batch, batch_size):
    return tf.reshape(
        tf.stack([v for _, v in board_tensors_batch.items()], axis=1),
        (batch_size, WIDTH, HEIGHT, CHANNELS, 1),
    )


def head_dataset(path, chunk=10):
    """For debugging. To use like:
    samples, board_tensors, labels, logs = head_dataset("data/mcts-playouts")
    """
    isamples = pd.read_csv(
        f"{path}/samples.csv.gzip", compression="gzip", iterator=True
    )
    iboard_tensors = pd.read_csv(
        f"{path}/board_tensors.csv.gzip", compression="gzip", iterator=True
    )
    ilabels = pd.read_csv(f"{path}/labels.csv.gzip", compression="gzip", iterator=True)
    ilogs = pd.read_csv(f"{path}/logs.csv.gzip", compression="gzip", iterator=True)

    return (
        isamples.get_chunk(chunk),
        iboard_tensors.get_chunk(chunk),
        ilabels.get_chunk(chunk),
        ilogs.get_chunk(chunk),
    )
