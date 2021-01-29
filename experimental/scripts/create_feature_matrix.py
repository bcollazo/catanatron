from pprint import pprint

import pandas as pd
import numpy as np
import click

from catanatron_server.database import get_finished_games_ids, get_last_game_state, get_game_states
from experimental.machine_learning.plot import plot_feature_importances, train
from experimental.machine_learning.features import feature_extractors, create_sample


@click.command()
@click.option("-n", "--number", default=5, help="Number of games for training.")
def create_feature_matrix(number):
    # Read games from game_states table and create samples.
    #   For each state: (p1, p2, p3, p4, winner), (p2, p3, p4, p1, winner), ...
    samples = []
    labels = []
    for game_id in get_finished_games_ids(limit=number):
        game = get_last_game_state(game_id)
        print(game_id, game)

        players = game.players
        winner = game.winning_player()
        if winner is None:
            print("SKIPPING NOT FINISHED GAME", game)
            continue

        label = game.players[0] == winner
        for state in get_game_states(game_id):
            samples.append(create_sample(game))
            labels.append(label)
            # for i, player in enumerate(players):
            #     p1, p2, p3 = [
            #         players[(i + 1) % len(players)],
            #         players[(i + 2) % len(players)],
            #         players[(i + 3) % len(players)],
            #     ]
            #     samples.append(create_sample(game, player, p1, p2, p3))

            # for i, player in enumerate(players):
            #     label = player == winner
            #     labels.append(label)

    print(len(samples))
    if len(samples) > 0:
        X = pd.DataFrame.from_records(samples)
        Y = np.array(labels)
        print(X.head())
        print(X.describe())
        pprint(sorted(X.columns))
        pprint(list(X.columns))

        # print("Training GreedyPlayer...")
        # train(X, Y)

        print("Plotting Feature Importances...")
        plot_feature_importances(X, Y)


if __name__ == "__main__":
    create_feature_matrix()
