import os
import uuid
import random
from pprint import pprint
import time
from collections import deque

# try to suppress TF output before any potentially tf-importing modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.progress import Progress
from rich.progress import Progress
from rich.console import Console
from rich.theme import Theme


from catanatron.game import Game
from catanatron_experimental.alphatan.mcts import AlphaMCTS, game_end_value
from catanatron_experimental.machine_learning.utils import ensure_dir
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron_experimental.play import play_batch
from catanatron_gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    from_action_space,
    to_action_space,
)
from catanatron_experimental.dqn_player import epsilon_greedy_policy
from catanatron_experimental.machine_learning.board_tensor_features import (
    CHANNELS,
    HEIGHT,
    WIDTH,
    create_board_tensor,
)
from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)

# For more repetitive results
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)

FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)
DATA_DIRECTORY = "data/simple-alphatan"
LOAD_MODEL = False

# 50 games per iteration, 10 iterations, 15 game pits takes ~8 hours.
ARGS = {
    "replay_memory_size": 200_000,
    "num_iterations": 10,
    "num_episodes_per_iteration": 50,
    "num_games_to_pit": 15,
    "model_acceptance_threshold": 0.55,
    "num_simulations_per_turn": 10,
    "temp_threshold": 30,
    "epochs": 10,
    "batch_size": 32,
}

custom_theme = Theme(
    {
        "progress.remaining": "",
        "progress.percentage": "",
        "bar.complete": "green",
        "bar.finished": "green",
    }
)
console = Console(theme=custom_theme)


def main():
    pprint(ARGS)
    replay_memory = deque(maxlen=ARGS["replay_memory_size"])
    flushable_samples = []
    ensure_dir(DATA_DIRECTORY)
    name = "alphatan"

    # initialize neural network
    model = create_model()
    if LOAD_MODEL:
        model.load_weights(f"data/checkpoints/{name}")

    # TODO: Load checkpoint
    # TODO: Simplify feature space. Only use board tensor and cards in hand (including dev cards).
    #   Has Road and Has Army.

    # for each iteration
    #   for each episode in iteration
    #     play game and produce many (s_t, policy_t, reward) samples.
    with Progress(console=console) as progress:
        main_task = progress.add_task(
            f'{ARGS["num_iterations"]} Iterations:', total=ARGS["num_iterations"]
        )
        for i in range(ARGS["num_iterations"]):
            episode_task = progress.add_task(
                f'{ARGS["num_episodes_per_iteration"]} Episodes:',
                total=ARGS["num_episodes_per_iteration"],
            )
            for _ in range(ARGS["num_episodes_per_iteration"]):
                examples = execute_episode(model)
                replay_memory += examples
                flushable_samples += examples

                progress.update(episode_task, advance=1)
            progress.update(main_task, advance=1)
            progress.remove_task(episode_task)

            print("Iteration:", i)
            print("Replay Memory Size:", len(replay_memory))
            print("Flushable Memory Size:", len(flushable_samples))

            # create new neural network from old one
            candidate_model = create_model()
            candidate_model.set_weights(model.get_weights())

            # write data for offline debugging.
            save_replay_memory(DATA_DIRECTORY, flushable_samples)
            flushable_samples = []

            # train new neural network
            print("Starting training...")
            start = time.time()
            random.shuffle(replay_memory)
            train(i, candidate_model, replay_memory)
            print("Done Training... Took", time.time() - start)

            # pit new vs old (with temp=0). If new wins, replace.
            model = pit(model, candidate_model)

            # TODO: Save checkpoint
            model.save_weights(f"data/checkpoints/{name}")
            model.save(f"data/models/{name}")


def save_replay_memory(data_directory, examples):
    print("Writing data...")
    states = pd.DataFrame([i for i in map(lambda x: x[0], examples)], columns=FEATURES)
    board_tensors = pd.DataFrame(
        [i for i in map(lambda x: np.array(x[1]).flatten(), examples)]
    )
    pis = pd.DataFrame(
        [i for i in map(lambda x: x[2], examples)],
        columns=[f"ACTION_{i}" for i in ACTIONS_ARRAY],
    )
    vs = pd.DataFrame([i for i in map(lambda x: x[3], examples)], columns=["V"])

    is_first_training = not os.path.isfile(
        os.path.join(data_directory, "states.csv.gzip")
    )
    states.to_csv(
        os.path.join(data_directory, "states.csv.gzip"),
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
    board_tensors.to_csv(
        os.path.join(data_directory, "board_tensors.csv.gzip"),
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
    pis.to_csv(
        os.path.join(data_directory, "pis.csv.gzip"),
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
    vs.to_csv(
        os.path.join(data_directory, "vs.csv.gzip"),
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
    print("Done Writing...")


def load_replay_memory(data_directory):
    states = pd.read_csv(
        os.path.join(data_directory, "states.csv.gzip"), compression="gzip"
    )
    board_tensors = pd.read_csv(
        os.path.join(data_directory, "board_tensors.csv.gzip"), compression="gzip"
    )
    pis = pd.read_csv(os.path.join(data_directory, "pis.csv.gzip"), compression="gzip")
    vs = pd.read_csv(os.path.join(data_directory, "vs.csv.gzip"), compression="gzip")
    return states, board_tensors, pis, vs


def pit(model, candidate_model, num_games=ARGS["num_games_to_pit"]):
    # Games seem to take ~16s (num_sim=10), so pitting 25 takes 6 mins.
    print("Starting Pit")
    players = [
        AlphaTan(Color.ORANGE, uuid.uuid4(), model, temp=0),
        AlphaTan(Color.WHITE, uuid.uuid4(), candidate_model, temp=0),
    ]
    wins, vp_history = play_batch(num_games, players)
    if wins[Color.WHITE] >= (sum(wins.values()) * ARGS["model_acceptance_threshold"]):
        print("Accepting model")
        return candidate_model
    else:
        print("Rejected model")
        return model


def execute_episode(model):
    # mcts = AlphaMCTS(model)  # so that its shared
    p0 = AlphaTan(Color.ORANGE, uuid.uuid4(), model)
    p1 = AlphaTan(Color.WHITE, uuid.uuid4(), model)
    game = Game(players=[p0, p1])
    game.play()

    winning_color = game.winning_color()
    if winning_color is None:
        return []

    examples = [
        (state, board_tensor, pi, 1 if color == winning_color else -1)
        for (color, state, board_tensor, pi) in p0.logs + p1.logs
    ]
    return examples


dfa = pd.read_csv("data/alphatan-memory/pis-mean.csv", index_col=0)


def train(iteration, model, replay_memory):
    input_boards, board_tensors, target_pis, target_vs = list(zip(*replay_memory))
    input_boards = np.asarray(input_boards)
    target_pis = np.asarray(target_pis)
    target_vs = np.asarray(target_vs)

    # Use sample_weight to lower the weights for END_TURN plays
    result = tf.linalg.normalize(target_pis, axis=0)
    class_weight = np.nan_to_num((1 / result[1]).numpy(), posinf=0)
    sample_weight = [class_weight[0][pi.argmax()] for pi in target_pis]

    model.fit(
        x=input_boards,
        y=[target_pis, target_vs],
        batch_size=ARGS["batch_size"],
        epochs=ARGS["epochs"],
        sample_weight=np.array(sample_weight),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=f"data/logs/alphatan-online/{iteration}",
                histogram_freq=1,
                write_graph=True,
            ),
        ],
    )


class AlphaTan(Player):
    def __init__(
        self,
        color,
        name,
        model,
        mcts=None,
        temp=None,
        num_simulations=ARGS["num_simulations_per_turn"],
    ):
        super().__init__(color)
        self.model = model
        self.mcts = mcts or AlphaMCTS(model)
        self.temp = temp
        self.logs = []

    def reset_state(self):
        self.mcts = AlphaMCTS(self.model)
        self.logs = []

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        start = time.time()

        for _ in range(ARGS["num_simulations_per_turn"]):
            self.mcts.search(game, self.color)

        # print("AlphaTan decision took", time.time() - start)
        # breakpoint()

        sample = create_sample_vector(game, self.color)
        board_tensor = create_board_tensor(game, self.color)
        s = tuple(sample)  # hashable s
        counts = [
            self.mcts.Nsa[(s, a)] if (s, a) in self.mcts.Nsa else 0
            for a in range(ACTION_SPACE_SIZE)
        ]

        # TODO: I think playable_actions is not needed b.c. search and counts force
        #   not-playable actions to have a 0.
        temp = (
            self.temp
            if self.temp is not None
            else int(game.state.num_turns < ARGS["temp_threshold"])
        )
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            pi = [0] * len(counts)
            pi[bestA] = 1
        else:
            counts = [x ** (1.0 / temp) for x in counts]
            counts_sum = float(sum(counts))
            pi = [x / counts_sum for x in counts]

        action = np.random.choice(len(pi), p=pi)
        best_action = from_action_space(action, game.state.playable_actions)
        self.logs.append((self.color, sample, board_tensor, pi))

        # if self.color == Color.WHITE:
        #     breakpoint()
        # print("AlphaTan decision took", time.time() - start)
        # breakpoint()

        return best_action


NEURONS_PER_LAYER = [32, 32, 32, 32, 32]


def create_model():
    inputs = tf.keras.Input(shape=(NUM_FEATURES,))
    outputs = inputs

    for neurons in NEURONS_PER_LAYER:
        outputs = tf.keras.layers.Dense(neurons, activation="relu")(outputs)

    pi_output = tf.keras.layers.Dense(
        ACTION_SPACE_SIZE,
        activation="softmax",
        # kernel_regularizer="l2",
    )(outputs)
    v_output = tf.keras.layers.Dense(1, activation="tanh")(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=[pi_output, v_output])
    model.compile(
        loss=["categorical_crossentropy", "mse"],
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    main()
