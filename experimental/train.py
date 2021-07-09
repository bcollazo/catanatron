"""
This script takes ReinforcementBatchPlayer and trains it further
by playing with itself more times. NOTE: Overwrites ReinforcementPlayer model.
"""
import os
import datetime

import click
import tensorflow as tf

from catanatron_gym.features import get_feature_ordering
from experimental.machine_learning.utils import (
    estimate_num_samples,
    generate_arrays_from_file,
    get_games_directory,
)
from experimental.machine_learning.players.reinforcement import (
    ACTION_SPACE_SIZE,
    p_model_path,
    q_model_path,
    v_model_path,
)

# TODO: Confirm "playouts" property. Load game. See value function. See playout distribution.
#       Is it repeatable / consistent?
# TODO: Speed up expansion features by caching 2 components.

BATCH_SIZE = 256
NUM_FEATURES = len(get_feature_ordering())


def train(
    games_directory,
    epochs,
    output_model_path,
    learning,
    label_threshold=None,
):
    if learning == "Q":
        shape = (NUM_FEATURES + ACTION_SPACE_SIZE,)
        output_units = 1
    elif learning == "V":
        shape = (NUM_FEATURES,)
        output_units = 1
    else:
        shape = (NUM_FEATURES,)
        output_units = ACTION_SPACE_SIZE

    tf.data.experimental.make_csv_dataset("data/random-games/samples.csv", 32)

    inputs = tf.keras.Input(shape=shape)
    outputs = inputs
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    # outputs = tf.keras.layers.Dropout(0.2)(outputs)
    # outputs = tf.keras.layers.Dense(64, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(256, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(128, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(64, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(units=output_units)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            "mean_absolute_error",
        ],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
        loss="mean_squared_error",
    )
    model.summary()

    # ===== V LEARNING
    label = "VICTORY_POINTS_RETURN"
    estimate = estimate_num_samples(games_directory)
    log_dir = f'logs/{learning}/fit/{label}/{estimate}/{epochs}/{BATCH_SIZE}/onelayer/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True
        )
    ]
    model.fit(
        generate_arrays_from_file(
            games_directory,
            BATCH_SIZE,
            label,
            learning=learning,
            label_threshold=label_threshold,
        ),
        steps_per_epoch=estimate / BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
    )

    if output_model_path is not None:
        model.save(output_model_path)
        print("Saved model at:", output_model_path)


@click.command()
@click.argument("player")
@click.option(
    "--epochs",
    default=10,
    help="Number of epochs for training.",
)
@click.option(
    "--data",
    default=None,
    help="Path to a games directory to use.",
)
@click.option(
    "--outpath",
    default=None,
    help="Path to write model.",
)
def cli(player, epochs, data, outpath):
    assert (
        len(player) == 2 and player[0] in ["V", "Q", "P"] and int(player[1]) >= 1
    ), "Player must be in format [V,Q,P][1-9]"

    learning = player[0]
    version = int(player[1])
    if version == 1:
        games_directory = get_games_directory()
    else:
        games_directory = get_games_directory(learning, version - 1)

    if data is not None:
        if os.path.normpath(games_directory) != os.path.normpath(data):
            print("WARNING: not using ", games_directory)
        games_directory = data

    if outpath is not None:
        if learning == "V":
            suggested_output_model_path = v_model_path(version)
        elif learning == "Q":
            suggested_output_model_path = q_model_path(version)
        else:
            suggested_output_model_path = p_model_path(version)
        if os.path.normpath(outpath) != os.path.normpath(suggested_output_model_path):
            print("WARNING: not using ", suggested_output_model_path)

    print(f"Training {learning}{version}. Using data from: {games_directory}.")
    print("Saving model to", outpath)

    if learning == "V":
        train(games_directory, epochs, outpath, learning=learning)
    elif learning == "Q":
        train(games_directory, epochs, outpath, learning=learning)
    else:
        train(
            games_directory,
            epochs,
            outpath,
            learning=learning,
            label_threshold=0.117294,
        )


# Model takes, input feature vector (~1000s features) and outputs ~5746 probability outputs.
# I am thinking several layers of 1500s hidden neurons.
# Q-Network S,A => %, needs 215,681 params
#              RETURN  DISCOUNTED_RETURN  TOURNAMENT_RETURN  DISCOUNTED_TOURNAMENT_RETURN
# count  72621.000000       72621.000000       72621.000000                  72621.000000
# mean       0.269123           0.089946         275.732171                     92.009600
# std        0.443507           0.165035         445.568912                    166.052533
# min        0.000000           0.000000           2.000000                      0.285411
# 25%        0.000000           0.000000           4.000000                      1.125086
# 50%        0.000000           0.000000           7.000000                      1.776989
# 75%        1.000000           0.117294        1010.000000                    118.466935
# max        1.000000           0.731867        1010.000000                    739.185945

if __name__ == "__main__":
    cli()
