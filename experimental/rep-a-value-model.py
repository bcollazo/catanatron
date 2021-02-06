import os
import random

import tensorflow as tf
import kerastuner as kt

from experimental.machine_learning.features import get_feature_ordering

# ===== Configuration
BATCH_SIZE = 256
EPOCHS = 1
PREFETCH_BUFFER_SIZE = 10
LABEL_COLUMN = "DISCOUNTED_RETURN"
LABEL_COLUMN = "VICTORY_POINTS_RETURN"
DATA_SIZE = 800 * 1000  # estimate: 800 samples per game.
DATA_DIRECTORY = "data/random-games"
STEPS_PER_EPOCH = DATA_SIZE / BATCH_SIZE
VALIDATION_DATA_SIZE = 800 * 10
VALIDATION_DATA_DIRECTORY = "data/validation-random-games"
SHUFFLE = True
SHUFFLE_SEED = random.randint(0, 20000)
SHUFFLE_BUFFER_SIZE = 1000
LOG_DIR = "./logs/rep-a-value-model"
NUM_FEATURES = len(get_feature_ordering())
MODEL_PATH = "experimental/models/dreturn-rep-a"


# ===== Building Dataset Generator
def preprocess(samples_batch, rewards_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_samples(samples_batch)
    label = rewards_batch[LABEL_COLUMN] / 10.0
    return (input1, label)


def preprocess_samples(samples_batch):
    return tf.stack([tensor for _, tensor in samples_batch.items()], axis=1)


def build_generator(dataset):
    def generator():
        for input1, label in dataset:
            yield input1, label

    return generator


print("Reading and building train dataset...")
samples = tf.data.experimental.make_csv_dataset(
    os.path.join(DATA_DIRECTORY, "samples.csv.gzip"),
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
train_dataset = tf.data.Dataset.zip((samples, rewards)).map(preprocess)
train_dataset = tf.data.Dataset.from_generator(
    build_generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)

print("Reading and building test dataset...")
samples = tf.data.experimental.make_csv_dataset(
    os.path.join(VALIDATION_DATA_DIRECTORY, "samples.csv.gzip"),
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
test_dataset = tf.data.Dataset.zip((samples, rewards)).map(preprocess)
test_dataset = tf.data.Dataset.from_generator(
    build_generator(test_dataset),
    output_signature=(
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)

# ===== REGULAR MODEL
inputs = tf.keras.Input(shape=(NUM_FEATURES,))
outputs = inputs
# outputs = tf.keras.layers.BatchNormalization()(outputs)
# outputs = tf.keras.layers.Dense(352, activation=tf.nn.relu)(outputs)
# outputs = tf.keras.layers.Dense(320, activation=tf.nn.relu)(outputs)
# outputs = tf.keras.layers.Dense(160, activation=tf.nn.relu)(outputs)
# outputs = tf.keras.layers.Dense(512, activation=tf.nn.relu)(outputs)
# outputs = tf.keras.layers.Dense(352, activation=tf.nn.relu)(outputs)
# outputs = tf.keras.layers.Dense(64, activation=tf.nn.relu)(outputs)
# outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(outputs)
outputs = tf.keras.layers.Dense(units=1)(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    metrics=["mae"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
    loss="mean_squared_error",
)
model.summary()

# ===== Keras-Tuners NN
# def model_builder(hp):
#     inputs = tf.keras.Input(shape=(NUM_FEATURES))
#     outputs = inputs
#     outputs = tf.keras.layers.Flatten()(outputs)
#     for i in range(hp.Int("num_layers", 2, 10)):
#         outputs = tf.keras.layers.Dense(
#             units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
#             activation="relu",
#         )(outputs)
#     outputs = tf.keras.layers.Dense(units=1)(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
#     model.compile(
#         metrics=["mae"],
#         optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#         loss="mse",
#     )
#     return model


# tuner = kt.Hyperband(
#     model_builder,
#     objective="val_mae",
#     max_epochs=EPOCHS,
#     factor=3,
#     directory="keras-tuner-models",
#     project_name="rep-a-value-model",
# )
# tuner.search(
#     train_dataset,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     epochs=EPOCHS,
#     validation_data=test_dataset,
#     validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=2, min_delta=1e-6),
#         tf.keras.callbacks.TensorBoard(
#             log_dir=LOG_DIR, histogram_freq=1, write_graph=True
#         ),
#     ],
# )
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"The hyperparameter search is complete.", best_hps)
# tuner.results_summary()
# model = tuner.hypermodel.build(best_hps)

# ===== Fit Final Model
model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=VALIDATION_DATA_SIZE / BATCH_SIZE,
    callbacks=[
        # tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, min_delta=1e-6),
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1, write_graph=True
        ),
    ],
)

model.save(MODEL_PATH)
print("Saved model at:", MODEL_PATH)
