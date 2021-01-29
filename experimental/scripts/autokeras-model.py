import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(os.environ["CUDA_VISIBLE_DEVICES"])

import autokeras as ak
import pandas as pd

directory = "data/random-games"
directory = "data/random-games-v1"
input_data = os.path.join(directory, "samples.csv")
label_data = os.path.join(directory, "rewards.csv")
# chunksize = 10 ** 6
# for chunk in pd.read_csv(input_data, chunksize=chunksize):
#     pass
print("Reading samples...")
tp = pd.read_csv(input_data, iterator=True, chunksize=1000)
X = pd.concat(tp, ignore_index=True)
print("Reading rewards...")
tp = pd.read_csv(label_data, iterator=True, chunksize=1000)
Y = pd.concat(tp, ignore_index=True)["VICTORY_POINTS_RETURN"]
# breakpoint()

# ===== Create AUTOKERAS Model
# Initialize the structured data classifier.
clf = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=3,
)  # It tries 3 different models.

# ===== Train KERAS Model
# Feed the structured data classifier with training data.
clf.fit(X, Y, epochs=10)

# ===== Test
# Evaluate the best model with testing data.
print("CLF SCORE =====")
print(clf.evaluate(X, Y))
# print(clf.predict(a))
model = clf.export_model()
model.save("ak-big-vptenth")
