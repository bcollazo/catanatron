import os
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
    get_tournament_total_return,
    get_victory_points_total_return,
    populate_matrices,
    simple_total_return,
)


class ReinforcementLearningAccumulator(GameAccumulator):
    def __init__(
        self,
        include_board_tensor=True,
        total_return_fns={
            "RETURN": simple_total_return,
            "TOURNAMENT_RETURN": get_tournament_total_return,
            "VICTORY_POINTS_RETURN": get_victory_points_total_return,
        },
    ):
        self.include_board_tensor = include_board_tensor
        # TODO: Generalize to "rewards_fn" that can yield intermediary rewards
        #   while still rewarding big on terminal states.
        self.total_return_fns = total_return_fns

    def before(self, game):
        self.data = {
            # e.g. {RED: [1,5]} if RED acted at tick 1 and 5
            "color_action_indices": defaultdict(list),
            "acting_color": [],
            "samples": [],
            "actions": [],
        }
        if self.include_board_tensor:
            self.data["board_tensors"] = []

    def step(self, game_before_action, action):
        self.data["color_action_indices"][action.color].append(
            len(self.data["samples"])
        )
        self.data["acting_color"].append(action.color)
        self.data["samples"].append(create_sample(game_before_action, action.color))
        self.data["actions"].append(
            [to_action_space(action), to_action_type_space(action.action_type)]
        )

        if self.include_board_tensor:
            board_tensor = create_board_tensor(game_before_action, action.color)
            flattened_tensor = board_tensor.reshape(-1)
            self.data["board_tensors"].append(flattened_tensor)

    def after(self, game):
        if game.winning_color() is None:
            return None  # drop game

        t1 = time.time()

        # Now that the game is over, we can calculate the returns
        # for each sample (so trajectories that lost still contribute data).
        returns = {
            name: np.zeros(len(self.data["samples"]), dtype=np.float64)
            for name in self.total_return_fns.keys()
        }
        for color, action_indices in self.data["color_action_indices"].items():
            # Set total return for the return of the perspective of this player
            player_returns = {
                name: np.full_like(
                    action_indices, total_return_fn(game, color), dtype=np.float64
                )
                for name, total_return_fn in self.total_return_fns.items()
            }

            # For each column, modify the indexes of this player
            for column_name, step_returns in player_returns.items():
                returns[column_name][action_indices] = step_returns

        T = len(self.data["samples"])
        discounts = DISCOUNT_FACTOR ** np.arange(T)[::-1]
        discount_columns = dict()
        for name, step_returns in returns.items():
            discount_columns["DISCOUNTED_" + name] = step_returns * discounts

        # Build Q-learning Design Matrix
        samples = self.data["samples"]
        actions = self.data["actions"]
        samples_df = (
            pd.DataFrame.from_records(samples, columns=sorted(samples[0].keys()))
            .astype("float64")
            .add_prefix("F_")
        )
        actions_df = pd.DataFrame(actions, columns=["ACTION", "ACTION_TYPE"]).astype(
            "int"
        )
        returns_df = pd.DataFrame({**returns, **discount_columns}).astype("float64")

        results = {
            "samples_df": samples_df,
            "actions_df": actions_df,
            "returns_df": returns_df,
        }
        if self.include_board_tensor:
            board_tensors = self.data["board_tensors"]
            board_tensors_df = (
                pd.DataFrame(board_tensors).astype("float64").add_prefix("BT_")
            )
            main_df = pd.concat(
                [samples_df, board_tensors_df, actions_df, returns_df], axis=1
            )
            results["board_tensors_df"] = board_tensors_df
            results["main_df"] = main_df
        else:
            main_df = pd.concat([samples_df, actions_df, returns_df], axis=1)
            results["main_df"] = main_df
        print(
            "Building matrices at took",
            format_secs(time.time() - t1),
        )
        return results


class CsvDataAccumulator(ReinforcementLearningAccumulator):
    def __init__(self, output, include_board_tensor=True):
        super().__init__(include_board_tensor)
        self.output = output

    def after(self, game):
        data = super().after(game)
        if data is None:
            return

        t1 = time.time()
        main_df = data["main_df"]
        samples_df = data["samples_df"]
        board_tensors_df = (
            None if not self.include_board_tensor else data["board_tensors_df"]
        )
        actions_df = data["actions_df"]
        returns_df = data["returns_df"]
        populate_matrices(
            samples_df,
            board_tensors_df,
            actions_df,
            returns_df,
            main_df,
            self.output,
        )
        print(
            f"Saved matrices to {self.output}{' (including board tensors)' if self.include_board_tensor else ''} with shapes: "
            f"main={main_df.shape}, samples={samples_df.shape}, actions={actions_df.shape}, "
            f"rewards={returns_df.shape} in {format_secs(time.time() - t1)}"
        )
        return samples_df, board_tensors_df, actions_df, returns_df


class ParquetDataAccumulator(ReinforcementLearningAccumulator):
    def __init__(self, output, include_board_tensor=True):
        super().__init__(include_board_tensor)
        self.output = output

    def after(self, game):
        data = super().after(game)
        if data is None:
            return

        t1 = time.time()
        main_df = data["main_df"]
        filepath = os.path.join(self.output, f"{game.id}.parquet")
        main_df.to_parquet(filepath, index=False)
        print(
            f"Saved main_df to {self.output} with shapes {main_df.shape} in {format_secs(time.time() - t1)}"
        )
        return main_df
