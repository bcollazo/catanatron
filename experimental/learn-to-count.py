# Import game data.
# Take layers that matter. Take only 10 samples.
# Take 2 for testing. Try to learn to count.

# Part 2: Count across layers.
# Part 3: Count only one layer ignore the other.

# from experimental.datasets import read_data

# data_directory = "data/random-games-separate-edge-nodes-simple-problem-big"
# samples, actions, rewards = read_data(data_directory)

from random import randint
import tensorflow as tf
import kerastuner as kt

WIDTH = 21
HEIGHT = 11
CHANNELS = 4


def gensample():
    plane = tf.zeros((WIDTH, HEIGHT))

    attempts = randint(1, WIDTH * HEIGHT)
    indices = list(
        set([(randint(0, WIDTH - 1), randint(0, HEIGHT - 1)) for _ in range(attempts)])
    )
    updates = [randint(1, 2) for _ in range(len(indices))]
    plane = tf.tensor_scatter_nd_update(plane, indices, updates)
    label = tf.reduce_sum(plane)

    random_plane1 = tf.zeros((WIDTH, HEIGHT))
    random_plane2 = tf.zeros((WIDTH, HEIGHT))
    random_plane3 = tf.zeros((WIDTH, HEIGHT))
    planes = tf.stack([plane, random_plane1, random_plane2, random_plane3], axis=2)
    return planes, label


data = []
labels = []
for i in range(100000):
    plane, label = gensample()
    data.append(plane)
    labels.append(label)
data = tf.convert_to_tensor(data)
labels = tf.convert_to_tensor(labels)

# ===== Basic NN
# inputs = tf.keras.Input(shape=(WIDTH, HEIGHT, CHANNELS))
# outputs = inputs
# outputs = tf.keras.layers.Flatten()(outputs)
# outputs = tf.keras.layers.Dense(units=32, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(units=32, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(units=1)(outputs)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.compile(
#     metrics=["mae"],
#     optimizer=tf.keras.optimizers.Adam(),
#     loss="mse",
# )

# ===== Basic CNN
inputs = tf.keras.Input(shape=(WIDTH, HEIGHT, CHANNELS))
outputs = inputs
outputs = tf.keras.layers.Conv2D(2, 3)(outputs)
outputs = tf.keras.layers.Flatten()(outputs)
outputs = tf.keras.layers.Dense(units=32, activation="relu")(outputs)
outputs = tf.keras.layers.Dense(units=1)(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    metrics=["mae"],
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse",
)

# ===== Keras-Tuners NN
# def model_builder(hp):
#     inputs = tf.keras.Input(shape=(WIDTH, HEIGHT, CHANNELS))
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


# tuner = kt.Hyperband(model_builder, objective="val_loss", max_epochs=10, factor=3)
# tuner.search(
#     x=data,
#     y=labels,
#     validation_split=0.1,
#     callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
# )
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"The hyperparameter search is complete.", best_hps)
# tuner.results_summary()
# model = tuner.hypermodel.build(best_hps)


model.summary()
model.fit(x=data, y=labels, validation_split=0.1, epochs=10)
breakpoint()
