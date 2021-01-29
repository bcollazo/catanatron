import tensorflow as tf

WIDTH = 16


def build_binary_data(a, b):
    data = []
    labels = []
    for i in range(a, b):
        binary_string = bin(i)[2:]
        binary_string = "".join(["0"] * (WIDTH - len(binary_string))) + binary_string
        data.append([int(x) for x in binary_string])
        labels.append(i % 2 == 1)
    return tf.convert_to_tensor(data), tf.convert_to_tensor(labels)


train_data, train_labels = build_binary_data(0, 2 ** WIDTH - 20)
test_data, test_labels = build_binary_data(2 ** WIDTH - 20, 2 ** WIDTH)

print("Building model...")
input_1 = tf.keras.layers.Input(shape=(WIDTH,))
x = input_1
# x = tf.keras.layers.Conv2D(1, kernel_size=3)(x)
# x = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=[input_1], outputs=x)


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

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-3, clipnorm=1.0),
    metrics=METRICS,
    loss="binary_crossentropy",
)
model.summary()

model.fit(train_data, train_labels)
print(model.predict(test_data) > 0.5)
breakpoint()


# This is for jupyterlab.
# class ClearTrainingOutput(tf.keras.callbacks.Callback):
#     def on_train_end(*args, **kwargs):
#         IPython.display.clear_output(wait=True)
