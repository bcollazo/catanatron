import os
import datetime

import tensorflow as tf
import kerastuner as kt

from experimental.machine_learning.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    NUMERIC_FEATURES,
    NUM_NUMERIC_FEATURES,
)

# ===== Configuration
BATCH_SIZE = 32
EPOCHS = 3
PREFETCH_BUFFER_SIZE = None
LABEL_COLUMN = "OWS_LABEL"
DATA_SIZE = 800 * 100  # estimate: 800 samples per game.
DATA_DIRECTORY = "data/random-games"
STEPS_PER_EPOCH = DATA_SIZE / BATCH_SIZE
VALIDATION_DATA_SIZE = 800 * 10
VALIDATION_DATA_DIRECTORY = "data/random-games"
SHUFFLE = True
SHUFFLE_SEED = 123
# CLASS_WEIGHT = {0: 0.05, 1: 0.95}
# CLASS_WEIGHT = None  # i really dont think this helps.

# ===== Building Dataset Generator
def preprocess(board_tensors_batch, rewards_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_board_tensors(board_tensors_batch)
    label = int(rewards_batch[LABEL_COLUMN])
    return (input1, label)


def preprocess_board_tensors(board_tensors_batch):
    return tf.reshape(
        tf.stack([v for k, v in board_tensors_batch.items()], axis=1),
        (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS),
    )


def preprocess_samples(samples_batch):
    return tf.stack(
        [
            tensor
            for feature_name, tensor in samples_batch.items()
            if feature_name in NUMERIC_FEATURES
        ],
        axis=1,
    )


def build_generator(dataset):
    def generator():
        for input1, label in dataset:
            breakpoint()
            yield input1, label

    return generator


print("Reading and building train dataset...")
board_tensors = tf.data.experimental.make_csv_dataset(
    os.path.join(DATA_DIRECTORY, "board_tensors.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
)
rewards = tf.data.experimental.make_csv_dataset(
    os.path.join(DATA_DIRECTORY, "rewards.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    select_columns=[LABEL_COLUMN],
)
train_dataset = tf.data.Dataset.zip((board_tensors, rewards)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, WIDTH, HEIGHT, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int8),
    ),
)

print("Reading and building test dataset...")
board_tensors = tf.data.experimental.make_csv_dataset(
    os.path.join(VALIDATION_DATA_DIRECTORY, "board_tensors.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
)
rewards = tf.data.experimental.make_csv_dataset(
    os.path.join(VALIDATION_DATA_DIRECTORY, "rewards.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    select_columns=[LABEL_COLUMN],
)
test_dataset = tf.data.Dataset.zip((board_tensors, rewards)).map(preprocess)
test_dataset = tf.data.Dataset.from_generator(
    build_generator(test_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, WIDTH, HEIGHT, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int8),
    ),
)


# ==== Multi-model
print("Building model...")

SHAPE = (WIDTH, HEIGHT, CHANNELS)
METRICS = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

# # the first branch operates on the first input
input_1 = tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS))
x = input_1
# x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(
    1,
    kernel_size=3,
    # activation="linear",
    # kernel_constraint=tf.keras.constraints.unit_norm(),
    # kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv2D(
#     32,
#     kernel_size=5,
#     activation="linear",
#     kernel_constraint=tf.keras.constraints.unit_norm(),
#     kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
# )(x)

x = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=[input_1], outputs=x)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-3, clipnorm=1.0),
    metrics=METRICS,
    loss="binary_crossentropy",
)
model.summary()


# def model_builder(hp):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Flatten(input_shape=SHAPE))

#     # Tune the number of units in the first Dense layer
#     # Choose an optimal value between 32-512
#     hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
#     hp_activation = hp.Choice("activation", values=["relu", "linear"])
#     model.add(tf.keras.layers.Dense(units=hp_units, activation=hp_activation))
#     model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # TODO:

#     # Tune the learning rate for the optimizer
#     # Choose an optimal value from 0.01, 0.001, or 0.0001
#     hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#         loss=tf.keras.losses.BinaryCrossentropy(),  # TODO:
#         metrics=METRICS,
#     )
#     return model


# tuner = kt.Hyperband(
#     model_builder,
#     objective="val_accuracy",
#     max_epochs=10,
#     factor=3,
#     directory="my_dir_shuffled",
#     project_name="intro_to_kt",
# )
# tuner.search(
#     train_dataset,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     epochs=EPOCHS,
#     validation_data=test_dataset,
#     validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
#     callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
# )
# # Get the optimal hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# print(
#     f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """
# )
# # Show a summary of the search
# tuner.results_summary()
# breakpoint()
# # Build the model with the optimal hyperparameters and train it on the data
# model = tuner.hypermodel.build(best_hps)


model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
)

# === Training
# print("Training...")
# print(
#     "Num Samples Estimate:",
#     DATA_SIZE,
#     "Steps per Epoch:",
#     steps_per_epoch,
#     "Epochs:",
#     EPOCHS,
# )
# log_dir = f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
# callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
# ]
# model.fit(
#     train_dataset,
#     steps_per_epoch=steps_per_epoch,
#     epochs=EPOCHS,
#     callbacks=callbacks,
#     class_weight=CLASS_WEIGHT,
# )

# # === Validation
# print("Doing Validation")
# model.evaluate(test_dataset, steps=VALIDATION_DATA_SIZE / BATCH_SIZE)


# # === Save image
# dot_img_file = "board-tensor-model.png"
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
# model.save("models/tensor-model-normalized")


from catanatron.models.player import Color
from catanatron_server.database import get_last_game_state
from experimental.machine_learning.features import create_sample, get_feature_ordering
from experimental.machine_learning.board_tensor_features import create_board_tensor
import tensorflow as tf


def predict(model, game_id, color):
    game = get_last_game_state(game_id)
    player = game.players_by_color[color]

    board_tensor = create_board_tensor(game, player)
    inputs1 = [board_tensor]

    sample = create_sample(game, player)
    input2 = [float(sample[i]) for i in get_feature_ordering() if i in NUMERIC_FEATURES]
    inputs2 = [input2]

    # return model.call([tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)])
    prediction = model.predict([tf.convert_to_tensor(inputs1)])
    # call = model.call([tf.convert_to_tensor(inputs1)])
    return prediction[0][0], prediction[0][0] > 0.5


print(predict(model, "d80ebf71-a1db-45b9-925a-2f4b95a28220", Color.BLUE), 1)
print(predict(model, "d80ebf71-a1db-45b9-925a-2f4b95a28220", Color.WHITE), 0)
print(predict(model, "d80ebf71-a1db-45b9-925a-2f4b95a28220", Color.ORANGE), 0)
print(predict(model, "d80ebf71-a1db-45b9-925a-2f4b95a28220", Color.RED), 1)

print(predict(model, "85825f88-38ad-443e-b6a8-7b8161b3bd6a", Color.BLUE), 1)
print(predict(model, "85825f88-38ad-443e-b6a8-7b8161b3bd6a", Color.WHITE), 0)
print(predict(model, "85825f88-38ad-443e-b6a8-7b8161b3bd6a", Color.ORANGE), 1)
print(predict(model, "85825f88-38ad-443e-b6a8-7b8161b3bd6a", Color.RED), 1)

print(predict(model, "944a2d03-926f-4e6f-865a-93c55d6103b7", Color.BLUE), 1)
print(predict(model, "944a2d03-926f-4e6f-865a-93c55d6103b7", Color.WHITE), 0)
print(predict(model, "944a2d03-926f-4e6f-865a-93c55d6103b7", Color.ORANGE), 0)
print(predict(model, "944a2d03-926f-4e6f-865a-93c55d6103b7", Color.RED), 1)
breakpoint()
print(model.get_weights())


# TODO: Evaluate bot by playing with randoms.
