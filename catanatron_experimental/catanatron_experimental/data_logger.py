from pathlib import Path

import pandas as pd

from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    CHANNELS,
    HEIGHT,
    WIDTH,
    create_board_tensor,
)


class DataLogger:
    """
    Class to accumulate states, write to CSV files, and read them to TF Datasets
    """

    def __init__(self, output_path):
        self.output_path = Path(output_path)

        self.samples = []
        self.board_tensors = []
        # TODO: Implement, Actions and Rewards
        self.labels = []
        self.log_lines = []

    def consume(self, game, mcts_labels):
        import tensorflow as tf  # lazy import tf so that catanatron simulator is usable without tf

        for color in game.state.colors:
            sample = create_sample_vector(game, color)
            flattened_board_tensor = tf.reshape(
                create_board_tensor(game, color),
                (WIDTH * HEIGHT * CHANNELS,),
            ).numpy()
            label = mcts_labels.get(color, 0)

            self.samples.append(sample)
            self.board_tensors.append(flattened_board_tensor)
            self.labels.append(label)
            self.log_lines.append(
                [
                    game.id,
                    len(game.state.actions),
                    "http://localhost:3000/games/" + game.id,
                ]
            )

    def get_replay_buffer(self):
        return self.samples, self.board_tensors, self.labels

    def flush(self):
        print("Flushing...")
        # Convert to dataframes for writing
        samples_df = pd.DataFrame(self.samples, columns=get_feature_ordering()).astype(
            "float64"
        )
        board_tensors_df = pd.DataFrame(self.board_tensors).astype("float64")
        labels_df = pd.DataFrame(self.labels).astype("float64")
        logs_df = pd.DataFrame(
            self.log_lines, columns=["GAME_ID", "LEN(GAME.ACTIONS)", "LINK"]
        )

        # Write to disk
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        samples_path = Path(self.output_path, "samples.csv.gzip")
        board_tensors_path = Path(self.output_path, "board_tensors.csv.gzip")
        labels_path = Path(self.output_path, "labels.csv.gzip")
        logs_path = Path(self.output_path, "logs.csv.gzip")

        is_first_training = not samples_path.is_file()
        samples_df.to_csv(
            samples_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        board_tensors_df.to_csv(
            board_tensors_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        labels_df.to_csv(
            labels_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        logs_df.to_csv(
            logs_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )

        # Flush Memory
        self.samples = []
        self.board_tensors = []
        self.labels = []
        print("Done flushing data")
