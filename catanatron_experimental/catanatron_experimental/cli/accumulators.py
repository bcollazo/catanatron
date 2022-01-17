import time
import os
import json
from collections import defaultdict

import tensorflow as tf
import numpy as np
import pandas as pd

from catanatron.game import Accumulator
from catanatron.json import GameEncoder
from catanatron.state_functions import get_actual_victory_points
from catanatron_server.models import database_session, upsert_game_state
from catanatron_server.utils import ensure_link
from catanatron_experimental.utils import formatSecs
from catanatron_experimental.machine_learning.utils import (
    get_discounted_return,
    get_tournament_return,
    get_victory_points_return,
    populate_matrices,
    DISCOUNT_FACTOR,
)
from catanatron_gym.features import create_sample
from catanatron_gym.envs.catanatron_env import to_action_space
from catanatron_experimental.machine_learning.board_tensor_features import (
    create_board_tensor,
)


class StatisticsAccumulator(Accumulator):
    def __init__(self):
        self.wins = defaultdict(int)
        self.turns = []
        self.ticks = []
        self.durations = []
        self.games = []
        self.results_by_player = defaultdict(list)

    def initialize(self, game):
        self.start = time.time()

    def finalize(self, game):
        duration = time.time() - self.start
        winning_color = game.winning_color()
        if winning_color is None:
            return  # do not track

        self.wins[winning_color] += 1
        self.turns.append(game.state.num_turns)
        self.ticks.append(len(game.state.actions))
        self.durations.append(duration)
        self.games.append(game)

        for player in game.state.players:
            points = get_actual_victory_points(game.state, player.color)
            self.results_by_player[player.color].append(points)

    def get_avg_ticks(self):
        return sum(self.ticks) / len(self.ticks)

    def get_avg_turns(self):
        return sum(self.turns) / len(self.turns)

    def get_avg_duration(self):
        return sum(self.durations) / len(self.durations)


class StepDatabaseAccumulator(Accumulator):
    """
    Saves a game state to database for each tick.
    Slows game ~1s per tick.
    """

    def initialize(self, game):
        with database_session() as session:
            upsert_game_state(game, session)

    def step(game):
        with database_session() as session:
            upsert_game_state(game, session)


class DatabaseAccumulator(Accumulator):
    """Saves last game state to database"""

    def finalize(self, game):
        self.link = ensure_link(game)


class JsonDataAccumulator(Accumulator):
    def __init__(self, output):
        self.output = output

    def finalize(self, game):
        filepath = os.path.join(self.output, f"{game.id}.json")
        with open(filepath, "w") as f:
            f.write(json.dumps(game, cls=GameEncoder))


class CsvDataAccumulator(Accumulator):
    def __init__(self, output):
        self.output = output

    def initialize(self, game):
        self.data = defaultdict(
            lambda: {"samples": [], "actions": [], "board_tensors": []}
        )

    def step(self, game, action):
        self.data[action.color]["samples"].append(create_sample(game, action.color))
        self.data[action.color]["actions"].append(to_action_space(action))
        board_tensor = create_board_tensor(game, action.color)
        shape = board_tensor.shape
        flattened_tensor = tf.reshape(
            board_tensor, (shape[0] * shape[1] * shape[2],)
        ).numpy()
        self.data[action.color]["board_tensors"].append(flattened_tensor)

    def finalize(self, game):
        if game.winning_color() is None:
            return  # drop game

        print("Flushing to matrices...")
        t1 = time.time()
        samples = []
        actions = []
        board_tensors = []
        labels = []
        for color in game.state.colors:
            player_data = self.data[color]

            # Make matrix of (RETURN, DISCOUNTED_RETURN, TOURNAMENT_RETURN, DISCOUNTED_TOURNAMENT_RETURN)
            episode_return = get_discounted_return(game, color, 1)
            discounted_return = get_discounted_return(game, color, DISCOUNT_FACTOR)
            tournament_return = get_tournament_return(game, color, 1)
            vp_return = get_victory_points_return(game, color)
            discounted_tournament_return = get_tournament_return(
                game, color, DISCOUNT_FACTOR
            )

            samples.extend(player_data["samples"])
            actions.extend(player_data["actions"])
            board_tensors.extend(player_data["board_tensors"])
            return_matrix = np.tile(
                [
                    [
                        episode_return,
                        discounted_return,
                        tournament_return,
                        discounted_tournament_return,
                        vp_return,
                    ]
                ],
                (len(player_data["samples"]), 1),
            )
            labels.extend(return_matrix)

        # Build Q-learning Design Matrix
        samples_df = (
            pd.DataFrame.from_records(samples, columns=sorted(samples[0].keys()))
            .astype("float64")
            .add_prefix("F_")
        )
        board_tensors_df = (
            pd.DataFrame(board_tensors).astype("float64").add_prefix("BT_")
        )
        actions_df = pd.DataFrame(actions, columns=["ACTION"]).astype("int")
        rewards_df = pd.DataFrame(
            labels,
            columns=[
                "RETURN",
                "DISCOUNTED_RETURN",
                "TOURNAMENT_RETURN",
                "DISCOUNTED_TOURNAMENT_RETURN",
                "VICTORY_POINTS_RETURN",
            ],
        ).astype("float64")
        main_df = pd.concat(
            [samples_df, board_tensors_df, actions_df, rewards_df], axis=1
        )

        print(
            "Collected DataFrames. Data size:",
            "Main:",
            main_df.shape,
            "Samples:",
            samples_df.shape,
            "Board Tensors:",
            board_tensors_df.shape,
            "Actions:",
            actions_df.shape,
            "Rewards:",
            rewards_df.shape,
        )
        populate_matrices(
            samples_df,
            board_tensors_df,
            actions_df,
            rewards_df,
            main_df,
            self.output,
        )
        print(
            "Saved to matrices at:",
            self.output,
            ". Took",
            formatSecs(time.time() - t1),
        )
        return samples_df, board_tensors_df, actions_df, rewards_df
