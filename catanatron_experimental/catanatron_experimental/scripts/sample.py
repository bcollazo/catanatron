import tensorflow as tf
import numpy as np

WIDTH = 21
HEIGHT = 11
CHANNELS = 16


def generator(N=10):
    """
    Returns tuple of (inputs,outputs) where
    inputs  = (inp1,inp2,inp2)
    outputs = out1
    """
    dt = np.float32
    for i in range(N):
        inputs = (
            np.random.rand(N, WIDTH, HEIGHT, CHANNELS).astype(dt),
            np.random.rand(N, 2).astype(dt),
        )
        outputs = np.random.rand(N, 1).astype(dt)
        yield inputs, outputs


# Create dataset from generator
data = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, WIDTH, HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    ),
)

# Define a model
inp1 = tf.keras.Input(shape=(WIDTH, HEIGHT, CHANNELS), name="inp1")
inp2 = tf.keras.Input(shape=(2,), name="inp2")
out1 = tf.keras.layers.Conv2D(1, kernel_size=3, padding="same")(inp1)
out1 = tf.keras.layers.Flatten()(out1)
out1 = tf.keras.layers.Dense(1)(out1)
model = tf.keras.Model(inputs=[inp1, inp2], outputs=out1)
model.compile(loss=["mse", "mse"])

# Train
model.fit(data)
