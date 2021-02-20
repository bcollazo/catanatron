from experimental.machine_learning.board_tensor_features import (
    CHANNELS,
    HEIGHT,
    NUM_NUMERIC_FEATURES,
    WIDTH,
)
from experimental.machine_learning.players.online_mcts_dqn import NUM_FEATURES
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
)
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam

from catanatron.models.player import Color
from experimental.machine_learning.features import create_sample, create_sample_vector
from experimental.datasets import (
    preprocess_samples,
    read_dataset,
    build_board_tensors_preprocess,
)
from catanatron_server.database import get_finished_games_ids, get_game_states

# Get last 100 games from database.
# print("Getting last 100 games")
# samples = []
# for game_id in get_finished_games_ids(limit=100):
#     for game in get_game_states(game_id):
#         p0 = next(player for player in game.players if player.color == Color.WHITE)
#         samples.append(create_sample_vector(game, p0))
# print("DONE")

# Ensure replay funcionality works.
def preprocess(samples_batch, rewards_batch):
    input2 = preprocess_samples(samples_batch)
    label = preprocess_samples(rewards_batch)
    return (input2, label)


def build_generator(dataset):
    def generator():
        for input2, label in dataset:
            yield input2, label

    return generator


# Ensure matches samples df.
data_directory = "data/mcts-playouts-labeling-2"
samples = read_dataset(f"{data_directory}/samples.csv.gzip")
# board_tensors = read_dataset(f"{data_directory}/board_tensors.csv.gzip")
# board_tensors_dataset = board_tensors.map(build_board_tensors_preprocess(32))
labels = read_dataset(f"{data_directory}/labels.csv.gzip")

# test_dataset = tf.data.Dataset.from_generator(
#     build_generator(test_dataset),
#     output_signature=(
#         (
#             tf.TensorSpec(shape=(None, WIDTH, HEIGHT, CHANNELS, 1), dtype=tf.float32),
#             tf.TensorSpec(shape=(None, NUM_NUMERIC_FEATURES), dtype=tf.float32),
#         ),
#         tf.TensorSpec(shape=(None,), dtype=tf.float32),
#     ),
# )
test_samples = samples
test_labels = labels
train_samples = samples
train_labels = labels

train_dataset = tf.data.Dataset.zip((train_samples, train_labels)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)
test_dataset = tf.data.Dataset.zip((test_samples, test_labels)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(test_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)
breakpoint()

# for batch in board_tensors_dataset.as_numpy_iterator():
#     breakpoint()

# ===== Model A
inputs = Input(shape=(NUM_FEATURES,))
outputs = inputs

outputs = Dense(64, activation="relu")(outputs)
outputs = Dense(64, activation="relu")(outputs)
outputs = Dense(64, activation="relu")(outputs)
outputs = Dense(64, activation="relu")(outputs)

# TODO: We may want to change infra to predict all 4 winning probas.
#   So that mini-max makes more sense? Enemies wont min you, they'll max
#   themselves.
outputs = Dense(units=1, activation="linear")(outputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
model.summary()

model.fit(x=train_dataset, validation_data=test_dataset)

# ===== Model B
# STRIDES = (2, 2, 1)
# input_1 = tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS, 1))
# x = input_1
# # NOTE: I think last stride doestn matter
# filters_1 = 128
# DROPOUT_RATE = 0.2
# x = tf.keras.layers.Conv3D(
#     filters_1,
#     kernel_size=(5, 3, CHANNELS),
#     strides=STRIDES,
#     data_format="channels_last",
#     activation="relu",
#     kernel_constraint=tf.keras.constraints.unit_norm(),
#     kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
# )(x)
# # x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
# # reshape so that we have filters_1 "channels" of the board, and we can do Conv2D
# x = tf.keras.layers.Reshape((9, 5, filters_1))(x)  # now each pixel is a tile
# # take tile triples hierarchy.
# x = tf.keras.layers.Conv2D(
#     1,
#     kernel_size=3,
#     data_format="channels_last",
#     activation="relu",
#     kernel_constraint=tf.keras.constraints.unit_norm(),
#     kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
# )(x)
# # x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.Model(inputs=input_1, outputs=x)

# input_2 = tf.keras.layers.Input(shape=(NUM_NUMERIC_FEATURES,))
# y = tf.keras.Model(inputs=input_2, outputs=input_2)

# combined = tf.keras.layers.concatenate([x.output, y.output])
# z = tf.keras.layers.Dense(32, activation="relu")(combined)
# # z = tf.keras.layers.Dense(32, activation="relu")(z)
# z = tf.keras.layers.Dense(1, activation="linear")(z)
# model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)
# model.compile(
#     metrics=["mae"],
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
#     loss="mean_squared_error",
# )
# model.summary()
