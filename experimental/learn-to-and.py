# Import game data.
# Take layers that matter. Take only 10 samples.
# Take 2 for testing. Try to learn to count.

# Part 2: Count across layers.
# Part 3: Count only one layer ignore the other.

# from experimental.datasets import read_data

# data_directory = "data/random-games-separate-edge-nodes-simple-problem-big"
# samples, actions, rewards = read_data(data_directory)

from random import randint, random, seed
import tensorflow as tf
import kerastuner as kt

WIDTH = 21
HEIGHT = 21
CHANNELS = 5

seed(123)


def gensample():
    plane1 = tf.zeros((WIDTH, HEIGHT))
    plane2 = tf.zeros((WIDTH, HEIGHT))

    if random() > 0.5:
        # make planes agree.
        indices = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1))]
        updates = [1]
        plane1 = tf.tensor_scatter_nd_update(plane1, indices, updates)
        plane2 = tf.tensor_scatter_nd_update(plane2, indices, updates)
        label = 1
    else:
        indices = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1))]
        indices2 = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1))]
        while indices2 == indices:
            indices2 = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1))]
        updates = [1]
        plane1 = tf.tensor_scatter_nd_update(plane1, indices, updates)
        plane2 = tf.tensor_scatter_nd_update(plane2, indices2, updates)
        label = 0

    random_plane1 = tf.zeros((WIDTH, HEIGHT))
    random_plane2 = tf.zeros((WIDTH, HEIGHT))
    random_plane3 = tf.zeros((WIDTH, HEIGHT))
    planes = tf.stack(
        [plane1, plane2, random_plane1, random_plane2, random_plane3], axis=2
    )
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
# outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(outputs)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.compile(
#     metrics=["accuracy"],
#     optimizer=tf.keras.optimizers.Adam(),
#     loss="binary_crossentropy",
# )

# ===== Basic CNN
inputs = tf.keras.Input(shape=(WIDTH, HEIGHT, CHANNELS))
outputs = inputs
outputs = tf.keras.layers.Conv2D(1, 1)(outputs)
outputs = tf.keras.layers.Flatten()(outputs)
outputs = tf.keras.layers.Dense(units=16, activation="relu")(outputs)
outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
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
#     outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
#     model.compile(
#         metrics=["accuracy"],
#         optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#         loss="binary_crossentropy",
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
# model = tuner.hypermodel.build(best_hps)


model.summary()
model.fit(
    x=data,
    y=labels,
    epochs=1000,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)],
)
breakpoint()
