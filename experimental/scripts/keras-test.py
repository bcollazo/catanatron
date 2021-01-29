import autokeras as ak
import numpy as np
import tensorflow as tf
import pandas as pd

input_data = "data/validation-data/samples.csv"
label_data = "data/validation-data/rewards.csv"
# chunksize = 10 ** 6
# for chunk in pd.read_csv(input_data, chunksize=chunksize):
#     pass
print("Reading samples...")
tp = pd.read_csv(input_data, iterator=True, chunksize=1000)
X = pd.concat(tp, ignore_index=True)
print("Reading rewards...")
tp = pd.read_csv(label_data, iterator=True, chunksize=1000)
Y = pd.concat(tp, ignore_index=True)["DISCOUNTED_RETURN"]
# breakpoint()


model = tf.keras.models.load_model(
    "model_autokeras_raw_features", custom_objects=ak.CUSTOM_OBJECTS
)
print("CLF SCORE =====")
# print(model.evaluate(X[:10], Y))
# breakpoint()
