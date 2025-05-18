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
    get_discounted_returns,
    get_tournament_total_return,
    get_victory_points_total_return,
    populate_matrices,
    return_to_rewards,
    simple_return,
)


class CsvDataAccumulator(GameAccumulator):
    def __init__(self, output):
        self.output = output

    def before(self, game):
        self.data = {
            # e.g. {RED: [1,5]} if RED acted at tick 1 and 5
            "color_action_indices": defaultdict(list),
            "acting_color": [],
            "samples": [],
            "board_tensors": [],
            "actions": [],
        }

    def step(self, game_before_action, action):
        self.data["color_action_indices"][action.color].append(
            len(self.data["samples"])
        )
        self.data["acting_color"].append(action.color)
        self.data["samples"].append(create_sample(game_before_action, action.color))
        self.data["actions"].append(
            [to_action_space(action), to_action_type_space(action)]
        )

        board_tensor = create_board_tensor(game_before_action, action.color)
        flattened_tensor = board_tensor.reshape(-1)
        self.data["board_tensors"].append(flattened_tensor)

    def after(self, game):
        if game.winning_color() is None:
            return  # drop game

        print("Flushing to matrices...")
        t1 = time.time()
        samples = self.data["samples"]
        actions = self.data["actions"]
        board_tensors = self.data["board_tensors"]

        # Get rewards vector. For now either -1 or 1.
        all_full_returns = np.zeros(len(self.data["samples"]))
        all_discounted_returns = np.zeros(len(self.data["samples"]))
        all_tournament_returns = np.zeros(len(self.data["samples"]))
        all_discounted_tournament_returns = np.zeros(len(self.data["samples"]))
        all_victory_points_returns = np.zeros(len(self.data["samples"]))
        all_discounted_victory_points_returns = np.zeros(len(self.data["samples"]))
        for color, action_indices in self.data["color_action_indices"].items():
            sparse_simple_rewards = return_to_rewards(
                simple_return(game, color), len(action_indices)
            )
            sparse_tournament_rewards = return_to_rewards(
                get_tournament_total_return(game, color), len(action_indices)
            )
            sparse_victory_points_rewards = return_to_rewards(
                get_victory_points_total_return(game, color), len(action_indices)
            )

            full_returns = get_discounted_returns(sparse_simple_rewards, 1)
            discounted_returns = get_discounted_returns(
                sparse_simple_rewards, DISCOUNT_FACTOR
            )
            full_tournament_returns = get_discounted_returns(
                sparse_tournament_rewards, 1
            )
            discounted_tournament_returns = get_discounted_returns(
                sparse_tournament_rewards, DISCOUNT_FACTOR
            )
            full_victory_points_returns = get_discounted_returns(
                sparse_victory_points_rewards, 1
            )
            discounted_victory_points_returns = get_discounted_returns(
                sparse_victory_points_rewards, DISCOUNT_FACTOR
            )

            all_full_returns[action_indices] = full_returns
            all_discounted_returns[action_indices] = discounted_returns
            all_tournament_returns[action_indices] = full_tournament_returns
            all_discounted_tournament_returns[action_indices] = (
                discounted_tournament_returns
            )
            all_victory_points_returns[action_indices] = full_victory_points_returns
            all_discounted_victory_points_returns[action_indices] = (
                discounted_victory_points_returns
            )

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
            {
                "RETURN": all_full_returns,
                "DISCOUNTED_RETURN": all_discounted_returns,
                "TOURNAMENT_RETURN": all_tournament_returns,
                "DISCOUNTED_TOURNAMENT_RETURN": all_discounted_tournament_returns,
                "VICTORY_POINTS_RETURN": all_victory_points_returns,
                "DISCOUNTED_VICTORY_POINTS_RETURN": all_discounted_victory_points_returns,
            }
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
