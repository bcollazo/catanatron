from pathlib import Path
import time
from pprint import pprint

import tensorflow as tf
import pickle
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from experimental.machine_learning.features import get_feature_ordering
from experimental.datasets import read_dataset, preprocess_samples

# Taken from correlation analysis
FEATURES = [
    "P0_HAS_ROAD",
    "P1_HAS_ROAD",
    "P0_HAS_ARMY",
    "P1_HAS_ARMY",
    "P0_ORE_PRODUCTION",
    "P0_WOOD_PRODUCTION",
    "P0_WHEAT_PRODUCTION",
    "P0_SHEEP_PRODUCTION",
    "P0_BRICK_PRODUCTION",
    "P0_LONGEST_ROAD_LENGTH",
    "P1_ORE_PRODUCTION",
    "P1_WOOD_PRODUCTION",
    "P1_WHEAT_PRODUCTION",
    "P1_SHEEP_PRODUCTION",
    "P1_BRICK_PRODUCTION",
    "P1_LONGEST_ROAD_LENGTH",
    "P0_PUBLIC_VPS",
    "P1_PUBLIC_VPS",
    "P0_SETTLEMENTS_LEFT",
    "P1_SETTLEMENTS_LEFT",
    "P0_CITIES_LEFT",
    "P1_CITIES_LEFT",
    "P0_KNIGHT_PLAYED",
    "P1_KNIGHT_PLAYED",
]
FEATURES = get_feature_ordering(num_players=2)


def read_data_tf():
    start = time.time()
    data_directory = "data/random1v1s"
    CHUNKS = 100_000
    samples = read_dataset(
        str(Path(data_directory, "samples.csv.gzip")),
        batch_size=1,
        shuffle_seed=1,
        shuffle_buffer_size=100_000,
    )
    samples_iter = samples.as_numpy_iterator()
    to_concat = []
    for i, sample in enumerate(samples_iter):
        df = pd.DataFrame(preprocess_samples(sample).numpy(), columns=sample.keys())
        to_concat.append(df)
        if i > CHUNKS:
            break
    samples_df = pd.concat(to_concat).reset_index(drop=True)

    rewards = read_dataset(
        str(Path(data_directory, "rewards.csv.gzip")),
        batch_size=1,
        shuffle_buffer_size=100_000,
        shuffle_seed=1,
        column_defaults=[tf.float64] * 10,
    )
    rewards_iter = rewards.as_numpy_iterator()
    to_concat = []
    for i, reward in enumerate(rewards_iter):
        df = pd.DataFrame(preprocess_samples(reward).numpy(), columns=reward.keys())
        to_concat.append(df)
        if i > CHUNKS:
            break
    rewards_df = pd.concat(to_concat).reset_index(drop=True)
    print("Reading data took:", time.time() - start)

    X = samples_df[FEATURES]
    y = rewards_df["DISCOUNTED_RETURN"]
    return X, y


def read_data_simple():
    #  zcat data/testing/rewards.csv.gzip | head -n 100000 > rewards.csv
    samples = pd.read_csv("samples.csv")[FEATURES]
    rewards = pd.read_csv("rewards.csv")["VICTORY_POINTS_RETURN"]
    return samples, rewards


if __name__ == "__main__":
    print("Reading RF data")
    X, y = read_data_simple()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    print("Training RF")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    print(model.feature_importances_)
    pprint(list(zip(FEATURES, model.feature_importances_)))
    print(model.predict([X_train.iloc[0]]))
    print(model.score(X_test, y_test))
    breakpoint()

    # Save model
    with open("experimental/models/simple-rf.model", "wb") as file:
        pickle.dump(model, file)
