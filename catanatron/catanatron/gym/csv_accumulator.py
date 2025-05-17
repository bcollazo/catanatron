from collections import defaultdict
import time

from catanatron.utils import format_secs
import numpy as np
import pandas as pd

from catanatron.features import create_sample
from catanatron.game import GameAccumulator
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.gym.envs.catanatron_env import to_action_space, to_action_type_space
from catanatron.gym.utils import (
    DISCOUNT_FACTOR,
    get_discounted_return,
    get_tournament_return,
    get_victory_points_return,
    populate_matrices,
)


class CsvDataAccumulator(GameAccumulator):
    def __init__(self, output):
        self.output = output

    def before(self, game):
        self.data = defaultdict(
            lambda: {"samples": [], "actions": [], "board_tensors": [], "games": []}
        )

    def step(self, game, action):
        self.data[action.color]["samples"].append(create_sample(game, action.color))
        self.data[action.color]["actions"].append(
            [to_action_space(action), to_action_type_space(action)]
        )
        self.data[action.color]["games"].append(game.copy())
        board_tensor = create_board_tensor(game, action.color)
        flattened_tensor = board_tensor.reshape(-1)
        self.data[action.color]["board_tensors"].append(flattened_tensor)

    def after(self, game):
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
            format_secs(time.time() - t1),
        )
        return samples_df, board_tensors_df, actions_df, rewards_df
