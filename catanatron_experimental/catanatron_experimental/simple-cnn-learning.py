import os
import random
import datetime

import tensorflow as tf
import kerastuner as kt

from catanatron_experimental.machine_learning.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    NUMERIC_FEATURES,
    NUM_NUMERIC_FEATURES,
)

# ===== Configuration
BATCH_SIZE = 256
EPOCHS = 2
PREFETCH_BUFFER_SIZE = 10
LABEL_COLUMN = "OWS_LABEL"
DATA_SIZE = 800 * 100  # estimate: 800 samples per game.
DATA_DIRECTORY = "data/random-games"
STEPS_PER_EPOCH = DATA_SIZE / BATCH_SIZE
VALIDATION_DATA_SIZE = 800 * 20
VALIDATION_DATA_DIRECTORY = "data/validation-random-games"
SHUFFLE = True
SHUFFLE_SEED = random.randint(0, 20000)
SHUFFLE_BUFFER_SIZE = 1000
INNER_CHANNELS = 13
LOG_DIR = "./logs/ows-label-kt-cnn"
# CLASS_WEIGHT = {0: 0.05, 1: 0.95}
# CLASS_WEIGHT = None  # i really dont think this helps.
print("SHUFFLE_SEED", SHUFFLE_SEED)

# ===== Building Dataset Generator
def preprocess(board_tensors_batch, rewards_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_board_tensors(board_tensors_batch)
    label = int(rewards_batch[LABEL_COLUMN])
    return (input1, label)


def preprocess_board_tensors(board_tensors_batch):
    tensor = tf.reshape(
        tf.stack([v for k, v in board_tensors_batch.items()], axis=1),
        (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS, 1),
    )
    return tensor[:, :, :, :13]  # only use buildings and resources


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
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
)
rewards = tf.data.experimental.make_csv_dataset(
    os.path.join(DATA_DIRECTORY, "rewards.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=[LABEL_COLUMN],
)
train_dataset = tf.data.Dataset.zip((board_tensors, rewards)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, WIDTH, HEIGHT, INNER_CHANNELS, 1), dtype=tf.float32),
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
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
)
rewards = tf.data.experimental.make_csv_dataset(
    os.path.join(VALIDATION_DATA_DIRECTORY, "rewards.csv.gzip"),
    BATCH_SIZE,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
    shuffle=SHUFFLE,
    shuffle_seed=SHUFFLE_SEED,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    select_columns=[LABEL_COLUMN],
)
test_dataset = tf.data.Dataset.zip((board_tensors, rewards)).map(preprocess)
test_dataset = tf.data.Dataset.from_generator(
    build_generator(test_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, WIDTH, HEIGHT, INNER_CHANNELS, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int8),
    ),
)


# ==== Multi-model
print("Building model...")
SHAPE = (WIDTH, HEIGHT, INNER_CHANNELS, 1)
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
# input_1 = tf.keras.layers.Input(shape=SHAPE)
# x = input_1
# # x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv3D(
#     32,
#     kernel_size=(5, 3, INNER_CHANNELS),
#     # activation="linear",
#     # kernel_constraint=tf.keras.constraints.unit_norm(),
#     # kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
# )(x)
# # x = tf.keras.layers.BatchNormalization()(x)
# # x = tf.keras.layers.Conv2D(
# #     32,
# #     kernel_size=5,
# #     activation="linear",
# #     kernel_constraint=tf.keras.constraints.unit_norm(),
# #     kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
# # )(x)

# # x = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
# x = tf.keras.layers.Flatten()(x)
# # x = tf.keras.layers.Dropout(0.4)(x)
# # x = tf.keras.layers.Dense(32, activation="relu")(x)
# # x = tf.keras.layers.Dropout(0.4)(x)
# x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
# model = tf.keras.Model(inputs=[input_1], outputs=x)
# model.compile(
#     # optimizer=tf.keras.optimizers.Adam(lr=1e-3, clipnorm=1.0),
#     optimizer="adam",
#     metrics=METRICS,
#     loss="binary_crossentropy",
# )
# model.summary()


def model_builder(hp):
    input_1 = tf.keras.layers.Input(shape=SHAPE)
    x = input_1

    # CNN Layers
    # last_filters = None
    # for i in range(hp.Int("num_conv_layers", 0, 3)):
    #     filters = hp.Int("filters_" + str(i), min_value=1, max_value=32, step=8)
    #     # filters = 32
    #     last_filters = filters
    filters = hp.Int("filters", min_value=1, max_value=32, step=8)
    x = tf.keras.layers.Conv3D(filters, kernel_size=(5, 3, INNER_CHANNELS))(x)

    # Deep Layer
    x = tf.keras.layers.Flatten()(x)
    for i in range(hp.Int("num_flat_layers", 0, 3)):
        units = hp.Int("units_" + str(i), min_value=8, max_value=32, step=8)
        x = tf.keras.layers.Dense(units=units, activation="relu")(x)

    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model = tf.keras.Model(inputs=[input_1], outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS,
    )
    return model


tuner = kt.Hyperband(
    model_builder,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="keras-tuner-models",
    project_name="simple-cnn",
)
tuner.search(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True
        ),
    ],
)
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(
    f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
"""
)
# Show a summary of the search
tuner.results_summary()
# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
    # callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True
        )
    ],
)


# #=== Save image
# dot_img_file = "board-tensor-model.png"
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
# model.save("models/tensor-model-normalized")
