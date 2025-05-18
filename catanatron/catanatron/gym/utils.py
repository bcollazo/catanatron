import os

import numpy as np

from catanatron.state_functions import get_actual_victory_points
from catanatron.utils import ensure_dir

# DISCOUNT_FACTOR = 0 would mean only focus on immediate reward. Must be < 1. The closer to 1, the more
#   important the future is. 0.99 means future is 100 times more important than immediate reward.
DISCOUNT_FACTOR = 0.99
DATA_DIRECTORY = "data"


def get_samples_path(games_directory):
    return os.path.join(games_directory, "samples.csv.gzip")


def get_board_tensors_path(games_directory):
    return os.path.join(games_directory, "board_tensors.csv.gzip")


def get_actions_path(games_directory):
    return os.path.join(games_directory, "actions.csv.gzip")


def get_rewards_path(games_directory):
    return os.path.join(games_directory, "rewards.csv.gzip")


def get_main_path(games_directory):
    return os.path.join(games_directory, "main.csv.gzip")


def get_matrices_path(games_directory):
    samples_path = get_samples_path(games_directory)
    board_tensors_path = get_board_tensors_path(games_directory)
    actions_path = get_actions_path(games_directory)
    rewards_path = get_rewards_path(games_directory)
    main_path = get_main_path(games_directory)
    return samples_path, board_tensors_path, actions_path, rewards_path, main_path


def get_games_directory(key=None, version=None):
    if key in set(["V", "P", "Q"]):
        return os.path.join(DATA_DIRECTORY, key, str(version))
    else:
        return os.path.join(DATA_DIRECTORY, "random_games")


def estimate_num_samples(games_directory):
    samples_path = get_samples_path(games_directory)
    file_size = os.path.getsize(samples_path)
    size_per_sample_estimate = 3906.25  # in bytes
    estimate = file_size // size_per_sample_estimate
    print(
        "Training via generator. File Size:",
        file_size,
        "Num Samples Estimate:",
        estimate,
    )
    return estimate


def generate_arrays_from_file(
    games_directory,
    batchsize,
    label_column,
    learning="Q",
    label_threshold=None,
):
    inputs = []
    targets = []
    batchcount = 0

    (
        samples_path,
        board_tensors_path,
        actions_path,
        rewards_path,
        main_path,
    ) = get_matrices_path(games_directory)
    while True:
        with open(samples_path) as s, open(actions_path) as a, open(rewards_path) as r:
            next(s)  # skip header
            next(a)  # skip header
            rewards_header = next(r)  # skip header
            label_index = rewards_header.rstrip().split(",").index(label_column)
            for i, sline in enumerate(s):
                try:
                    srecord = sline.rstrip().split(",")
                    arecord = a.readline().rstrip().split(",")
                    rrecord = r.readline().rstrip().split(",")

                    state = [float(n) for n in srecord[:]]
                    action = [float(n) for n in arecord[:]]
                    reward = float(rrecord[label_index])
                    if label_threshold is not None and reward < label_threshold:
                        continue

                    if learning == "Q":
                        sample = state + action
                        label = reward
                    elif learning == "V":
                        sample = state
                        label = reward
                    else:  # learning == "P"
                        sample = state
                        label = action

                    inputs.append(sample)
                    targets.append(label)
                    batchcount += 1
                except Exception as e:
                    print(i)
                    print(s)
                    print(e)
                if batchcount > batchsize:
                    X = np.array(inputs, dtype="float32")
                    y = np.array(targets, dtype="float32")
                    yield (X, y)
                    inputs = []
                    targets = []
                    batchcount = 0


def simple_return(game, color):
    """
    Get the final return for the given color.
    Args:
        game: The game object.
        color: The color of the player.
    Returns:
        float: The final return.
    """
    if game.winning_color() == color:
        return 1.0
    elif game.winning_color() is None:
        return 0.0
    else:
        return -1.0


def return_to_rewards(return_value, n):
    """
    Convert a return value to a rewards vector.
    """
    rewards = np.zeros(n)
    rewards[-1] = return_value
    return rewards


def get_tournament_total_return(game, p0_color):
    """
    Winning is worth 1000 points, and the number of victory points
    is worth 1 point. The factor (0.9999) ensures a game
    won in less turns is better, and still a Game with 9vps is less
    than 10vps, no matter turns.
    """
    sign = simple_return(game, p0_color)
    points = get_actual_victory_points(game.state, p0_color)
    return sign * 1000 + min(points, 10) * 0.9999**game.state.num_turns


def get_victory_points_total_return(game, p0_color):
    """
    The final reward will be the number of victory points, no matter
    if the game is won or not.
    """
    # This discount factor (0.9999) ensures a game won in less turns
    #   is better, and still a Game with 9vps is less than 10vps,
    #   no matter turns.
    points = get_actual_victory_points(game.state, p0_color)
    episode_return = min(points, 10)
    return episode_return * 0.9999**game.state.num_turns


def get_discounted_returns(rewards, gamma):
    """
    Compute discounted returns G_t for each timestep.
    Args:
        rewards (np.ndarray): Array of rewards [r_0, ..., r_T]
            if sparse rewards, most should be 0, except for the last one
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        np.ndarray: Discounted return G_t for each timestep t
    """
    T = len(rewards)

    rewards = np.array(rewards, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def populate_matrices(
    samples_df, board_tensors_df, actions_df, rewards_df, main_df, games_directory
):
    (
        samples_path,
        board_tensors_path,
        actions_path,
        rewards_path,
        main_path,
    ) = get_matrices_path(games_directory)

    ensure_dir(games_directory)

    is_first_training = not os.path.isfile(samples_path)
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
    actions_df.to_csv(
        actions_path,
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
    rewards_df.to_csv(
        rewards_path,
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
    main_df.to_csv(
        main_path,
        mode="a",
        header=is_first_training,
        index=False,
        compression="gzip",
    )
