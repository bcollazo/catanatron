import time

import pickle
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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


if __name__ == "__main__":
    samples = 300_000
    chunk = 1000

    # 300,000 samples amount about 5Gb of data
    start = time.time()
    path = "data/mcts-playouts"
    X = pd.read_csv(f"{path}/samples.csv.gzip", compression="gzip")
    Y = pd.read_csv(f"{path}/labels.csv.gzip", compression="gzip")
    X = X.loc[0:, FEATURES]
    Y = Y.loc[0:]
    print("Reading data took:", time.time() - start)
    # breakpoint()

    clf = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))
    clf.fit(X, Y.to_numpy().flatten())

    # for i in range(0, samples, chunk):
    #     x = X.get_chunk(chunk)
    #     y = Y.get_chunk(chunk).to_numpy().flatten()
    #     clf.partial_fit(x, y)

    print(clf.predict(X[0:10]), Y[0:10])
    print(clf.score(X[0:10], Y[0:10].to_numpy().flatten()))
    print(clf.score(X, Y.to_numpy().flatten()))

    # Save model
    with open("experimental/models/simple-scikit.model", "wb") as file:
        pickle.dump(clf, file)
