from pathlib import Path
import time
from pprint import pprint

import pickle
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from catanatron_gym.features import get_feature_ordering
from experimental.datasets import read_dataset, preprocess_samples


# Taken from correlation analysis
FEATURES = [
    "P0_HAS_ROAD",
    "P1_HAS_ROAD",
    "P2_HAS_ROAD",
    "P3_HAS_ROAD",
    "P0_HAS_ARMY",
    "P1_HAS_ARMY",
    "P2_HAS_ARMY",
    "P3_HAS_ARMY",
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
    "P2_ORE_PRODUCTION",
    "P2_WOOD_PRODUCTION",
    "P2_WHEAT_PRODUCTION",
    "P2_SHEEP_PRODUCTION",
    "P2_BRICK_PRODUCTION",
    "P2_LONGEST_ROAD_LENGTH",
    "P3_ORE_PRODUCTION",
    "P3_WOOD_PRODUCTION",
    "P3_WHEAT_PRODUCTION",
    "P3_SHEEP_PRODUCTION",
    "P3_BRICK_PRODUCTION",
    "P3_LONGEST_ROAD_LENGTH",
    "P0_PUBLIC_VPS",
    "P1_PUBLIC_VPS",
    "P2_PUBLIC_VPS",
    "P3_PUBLIC_VPS",
    "P0_SETTLEMENTS_LEFT",
    "P1_SETTLEMENTS_LEFT",
    "P2_SETTLEMENTS_LEFT",
    "P3_SETTLEMENTS_LEFT",
    "P0_CITIES_LEFT",
    "P1_CITIES_LEFT",
    "P2_CITIES_LEFT",
    "P3_CITIES_LEFT",
    "P0_KNIGHT_PLAYED",
    "P1_KNIGHT_PLAYED",
    "P2_KNIGHT_PLAYED",
    "P3_KNIGHT_PLAYED",
]
FEATURES = [
    "P0_HAS_ROAD",
    "P1_HAS_ROAD",
    "P0_HAS_ARMY",
    "P1_HAS_ARMY",
    "TOTAL_P0_ORE_PRODUCTION",
    "TOTAL_P0_WOOD_PRODUCTION",
    "TOTAL_P0_WHEAT_PRODUCTION",
    "TOTAL_P0_SHEEP_PRODUCTION",
    "TOTAL_P0_BRICK_PRODUCTION",
    "P0_LONGEST_ROAD_LENGTH",
    # "EFFECTIVE_P1_ORE_PRODUCTION",
    # "EFFECTIVE_P1_WOOD_PRODUCTION",
    # "EFFECTIVE_P1_WHEAT_PRODUCTION",
    # "EFFECTIVE_P1_SHEEP_PRODUCTION",
    # "EFFECTIVE_P1_BRICK_PRODUCTION",
    "P1_LONGEST_ROAD_LENGTH",
    "P0_PUBLIC_VPS",
    "P1_PUBLIC_VPS",
    "P0_SETTLEMENTS_LEFT",
    "P1_SETTLEMENTS_LEFT",
    "P0_CITIES_LEFT",
    "P1_CITIES_LEFT",
    "P0_KNIGHT_PLAYED",
    "P1_KNIGHT_PLAYED",
    # TODO: Production Points
    # TODO: Resource Variety
    # TODO: Cards in hand
    # TODO: Wood-Brick Ratio (Non-linear...)
    # TODO: Wheat-Ore Ratio (Non-linear...)
]


LABEL = "VICTORY_POINTS_RETURN"


def create_datasets_iters(data_directory):
    samples = read_dataset(
        str(Path(data_directory, "samples.csv.gzip")),
        batch_size=1,
        shuffle=False,
        select_columns=FEATURES,
    )
    samples_iter = samples.as_numpy_iterator()
    rewards = read_dataset(
        str(Path(data_directory, "rewards.csv.gzip")),
        batch_size=1,
        shuffle=False,
        select_columns=[LABEL],
        column_defaults=[tf.float64],
    )
    rewards_iter = rewards.as_numpy_iterator()

    return samples_iter, rewards_iter


def read_data_tf(samples_iter, rewards_iter, batches=1_000):
    start = time.time()

    to_concat = []
    for i, sample in enumerate(samples_iter):
        df = pd.DataFrame(preprocess_samples(sample).numpy(), columns=sample.keys())
        to_concat.append(df)
        if i > batches:
            break
    samples_df = pd.concat(to_concat).reset_index(drop=True)

    to_concat = []
    for i, reward in enumerate(rewards_iter):
        df = pd.DataFrame(preprocess_samples(reward).numpy(), columns=reward.keys())
        to_concat.append(df)
        if i > batches:
            break
    rewards_df = pd.concat(to_concat).reset_index(drop=True)
    print(f"Read {batches} samples. Took:", time.time() - start)

    return samples_df, rewards_df


def read_and_train_simple():
    #  zcat data/testing/rewards.csv.gzip | head -n 100000 > rewards.csv
    start = time.time()
    samples_df = pd.read_csv("samples.csv")
    rewards_df = pd.read_csv("rewards.csv")
    print("Reading data took:", time.time() - start)

    X = samples_df[FEATURES]
    y = rewards_df["VICTORY_POINTS_RETURN"]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.10, random_state=42
    )
    clf = make_pipeline(StandardScaler(), SGDRegressor())

    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(y_test)
    print(clf.score(X_test, y_test))

    pprint(list(zip(FEATURES, clf.steps[1][1].coef_)))

    return clf


def read_and_train_online():
    global_scaler = StandardScaler()
    clf = SGDRegressor()
    samples_iter, rewards_iter = create_datasets_iters("data/linear-features")
    for epochs in range(100):
        samples_df, rewards_df = read_data_tf(samples_iter, rewards_iter, 100_000)

        X = samples_df[FEATURES]
        y = rewards_df["VICTORY_POINTS_RETURN"]

        scaler = StandardScaler()
        scaler.fit(X)
        global_scaler.partial_fit(X)
        X = scaler.transform(X)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.10, random_state=42
        )

        clf.partial_fit(X_train, y_train)

        print(clf.predict(X_test))
        print(y_test)
        print(clf.score(X_test, y_test))

    pprint(list(zip(FEATURES, clf.coef_)))
    clf = make_pipeline(scaler, clf)
    return clf


if __name__ == "__main__":
    # clf = read_and_train_online()
    clf = read_and_train_simple()

    breakpoint()

    # Save model
    with open("experimental/models/simple-scikit-linear.model", "wb") as file:
        pickle.dump(clf, file)
