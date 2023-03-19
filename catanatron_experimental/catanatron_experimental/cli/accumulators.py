import time
import os
import json
from collections import defaultdict

import numpy as np

from catanatron.game import GameAccumulator, Game
from catanatron.json import GameEncoder
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_player_buildings,
)
from catanatron.models.enums import VICTORY_POINT, SETTLEMENT, CITY
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
from catanatron_gym.envs.catanatron_env import to_action_space, to_action_type_space
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
)


class VpDistributionAccumulator(GameAccumulator):
    """
    Accumulates CITIES,SETTLEMENTS,DEVVPS,LONGEST,LARGEST
    in each game per player.
    """

    def __init__(self):
        # These are all per-player. e.g. self.cities['RED']
        self.cities = defaultdict(int)
        self.settlements = defaultdict(int)
        self.devvps = defaultdict(int)
        self.longest = defaultdict(int)
        self.largest = defaultdict(int)

        self.num_games = 0

    def after(self, game: Game):
        winner = game.winning_color()
        if winner is None:
            return  # throw away data

        for color in game.state.colors:
            cities = len(get_player_buildings(game.state, color, CITY))
            settlements = len(get_player_buildings(game.state, color, SETTLEMENT))
            longest = get_longest_road_color(game.state) == color
            largest = get_largest_army(game.state)[0] == color
            devvps = get_dev_cards_in_hand(game.state, color, VICTORY_POINT)

            self.cities[color] += cities
            self.settlements[color] += settlements
            self.longest[color] += longest
            self.largest[color] += largest
            self.devvps[color] += devvps

        self.num_games += 1

    def get_avg_cities(self, color=None):
        if color is None:
            return sum(self.cities.values()) / self.num_games
        else:
            return self.cities[color] / self.num_games

    def get_avg_settlements(self, color=None):
        if color is None:
            return sum(self.settlements.values()) / self.num_games
        else:
            return self.settlements[color] / self.num_games

    def get_avg_longest(self, color=None):
        if color is None:
            return sum(self.longest.values()) / self.num_games
        else:
            return self.longest[color] / self.num_games

    def get_avg_largest(self, color=None):
        if color is None:
            return sum(self.largest.values()) / self.num_games
        else:
            return self.largest[color] / self.num_games

    def get_avg_devvps(self, color=None):
        if color is None:
            return sum(self.devvps.values()) / self.num_games
        else:
            return self.devvps[color] / self.num_games


class StatisticsAccumulator(GameAccumulator):
    def __init__(self):
        self.wins = defaultdict(int)
        self.turns = []
        self.ticks = []
        self.durations = []
        self.games = []
        self.results_by_player = defaultdict(list)

    def before(self, game):
        self.start = time.time()

    def after(self, game):
        duration = time.time() - self.start
        winning_color = game.winning_color()
        if winning_color is None:
            return  # do not track

        self.wins[winning_color] += 1
        self.turns.append(game.state.num_turns)
        self.ticks.append(len(game.state.actions))
        self.durations.append(duration)
        self.games.append(game)

        for color in game.state.colors:
            points = get_actual_victory_points(game.state, color)
            self.results_by_player[color].append(points)

    def get_avg_ticks(self):
        return sum(self.ticks) / len(self.ticks)

    def get_avg_turns(self):
        return sum(self.turns) / len(self.turns)

    def get_avg_duration(self):
        return sum(self.durations) / len(self.durations)


class StepDatabaseAccumulator(GameAccumulator):
    """
    Saves a game state to database for each tick.
    Slows game ~1s per tick.
    """

    def before(self, game):
        with database_session() as session:
            upsert_game_state(game, session)

    def step(self, game):
        with database_session() as session:
            upsert_game_state(game, session)


class DatabaseAccumulator(GameAccumulator):
    """Saves last game state to database"""

    def after(self, game):
        self.link = ensure_link(game)


class JsonDataAccumulator(GameAccumulator):
    def __init__(self, output):
        self.output = output

    def after(self, game):
        filepath = os.path.join(self.output, f"{game.id}.json")
        with open(filepath, "w") as f:
            f.write(json.dumps(game, cls=GameEncoder))


class CsvDataAccumulator(GameAccumulator):
    def __init__(self, output):
        self.output = output

    def before(self, game):
        self.data = defaultdict(
            lambda: {"samples": [], "actions": [], "board_tensors": [], "games": []}
        )

    def step(self, game, action):
        import tensorflow as tf  # lazy import tf so that catanatron simulator is usable without tf

        self.data[action.color]["samples"].append(create_sample(game, action.color))
        self.data[action.color]["actions"].append(
            [to_action_space(action), to_action_type_space(action)]
        )
        self.data[action.color]["games"].append(game.copy())
        board_tensor = create_board_tensor(game, action.color)
        shape = board_tensor.shape
        flattened_tensor = tf.reshape(
            board_tensor, (shape[0] * shape[1] * shape[2],)
        ).numpy()
        self.data[action.color]["board_tensors"].append(flattened_tensor)

    def after(self, game):
        import pandas as pd

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
            # TODO: return label, 2-ply search label, 1-play value function.

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
        actions_df = pd.DataFrame(actions, columns=["ACTION", "ACTION_TYPE"]).astype(
            "int"
        )
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
