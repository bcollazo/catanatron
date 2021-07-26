import datetime

import tensorflow as tf

from catanatron_experimental.machine_learning.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    NUMERIC_FEATURES,
    NUM_NUMERIC_FEATURES,
)

# ===== Configuration
NUM_SAMPLES_ESTIMATE = 1000
BATCH_SIZE = 256
EPOCHS = 1
PREFETCH_BUFFER_SIZE = None
NORMALIZATION_BATCHES = NUM_SAMPLES_ESTIMATE // BATCH_SIZE // 10
# LABEL_COLUMN = "VICTORY_POINTS_RETURN"
LABEL_COLUMN = "DISCOUNTED_RETURN"

# ===== Read dataset
print("Reading CSVs...")
samples = tf.data.experimental.make_csv_dataset(
    "data/random-games/samples.csv.gzip",
    BATCH_SIZE,
    shuffle=False,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
)
board_tensors = tf.data.experimental.make_csv_dataset(
    "data/random-games/board_tensors.csv.gzip",
    BATCH_SIZE,
    shuffle=False,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
)
rewards = tf.data.experimental.make_csv_dataset(
    "data/random-games/rewards.csv.gzip",
    BATCH_SIZE,
    shuffle=False,
    prefetch_buffer_size=PREFETCH_BUFFER_SIZE,
    compression_type="GZIP",
)


# ===== Building Dataset Generator
def preprocess(samples_batch, board_tensors_batch, rewards_batch):
    """Input are dictionary of tensors"""
    input1 = preprocess_board_tensors(board_tensors_batch)
    input2 = preprocess_samples(samples_batch)
    label = rewards_batch[LABEL_COLUMN]
    return (input1, input2, label)


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
        for input1, input2, label in dataset:
            yield (input1, input2), label

    return generator


print("Reshaping and building dataset...")
dataset = tf.data.Dataset.zip((samples, board_tensors, rewards)).map(preprocess)
dataset = tf.data.Dataset.from_generator(
    build_generator(dataset),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, WIDTH, HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_NUMERIC_FEATURES), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)

# print(f"Creating normalization layers with {NORMALIZATION_BATCHES} batches of data...")
# t1 = time.time()
# input1_normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
# input1_normalizer.adapt(
#     board_tensors.take(NORMALIZATION_BATCHES).map(preprocess_board_tensors)
# )
# input2_normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
# input2_normalizer.adapt(samples.take(NORMALIZATION_BATCHES).map(preprocess_samples))
# print(input1_normalizer.mean, input1_normalizer.variance)
# print("Took", time.time() - t1)


# ==== Multi-model
print("Building model...")
# the first branch operates on the first input
input_1 = tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS))
# x = input1_normalizer(input_1)
x = tf.keras.layers.BatchNormalization()(input_1)
x = tf.keras.layers.Conv2D(1, kernel_size=(3, 5), activation="linear")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.Model(inputs=input_1, outputs=x)

# the second branch opreates on the second input
input_2 = tf.keras.layers.Input(shape=(NUM_NUMERIC_FEATURES,))
# y = input2_normalizer(input_2)
# y = tf.keras.layers.Dense(32, activation="relu")(input_2)
y = tf.keras.Model(inputs=input_2, outputs=input_2)

# combine the output of the two branches
combined = tf.keras.layers.concatenate([x.output, y.output])
z = tf.keras.layers.Dense(64, activation="relu")(combined)
z = tf.keras.layers.Dense(32, activation="relu")(z)
z = tf.keras.layers.Dense(1, activation="linear")(z)
model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)
model.compile(
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(),
        "mean_absolute_error",
    ],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
    loss="mean_squared_error",
)
model.summary()

# === Training
print("Training...")
steps_per_epoch = NUM_SAMPLES_ESTIMATE / BATCH_SIZE
print(
    "Num Samples Estimate:",
    NUM_SAMPLES_ESTIMATE,
    "Steps per Epoch:",
    steps_per_epoch,
    "Epochs:",
    EPOCHS,
)
log_dir = f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
]
model.fit(
    dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# === Save image
dot_img_file = "board-tensor-model.png"
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
model.save("models/tensor-model-normalized")
