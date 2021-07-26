from pprint import pprint

import pickle
import sklearn
from sklearn.ensemble import RandomForestRegressor

from catanatron_experimental.simple_model import create_datasets_iters, read_data_tf
from catanatron_gym.features import get_feature_ordering

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


if __name__ == "__main__":
    samples_iter, rewards_iter = create_datasets_iters("data/random1v1s")
    for epochs in range(10):
        samples_df, rewards_df = read_data_tf(samples_iter, rewards_iter, 1000)

        X = samples_df[FEATURES]
        y = rewards_df["VICTORY_POINTS_RETURN"]

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
    with open("data/models/simple-rf.model", "wb") as file:
        pickle.dump(model, file)
