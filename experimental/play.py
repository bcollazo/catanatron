from experimental.machine_learning.players.scikit import ScikitPlayer
import re
import traceback
import time
from collections import defaultdict

import click
import termplotlib as tpl
import numpy as np
import pandas as pd
import tensorflow as tf

from catanatron.models.enums import BuildingType, Resource
from catanatron_server.database import save_game_state
from catanatron.game import Game
from catanatron.models.player import HumanPlayer, RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from experimental.machine_learning.players.reinforcement import (
    QRLPlayer,
    TensorRLPlayer,
    VRLPlayer,
    PRLPlayer,
    hot_one_encode_action,
)
from experimental.machine_learning.players.minimax import (
    MiniMaxPlayer,
    ValueFunctionPlayer,
)
from experimental.machine_learning.players.mcts import MCTSPlayer
from experimental.machine_learning.players.online_mcts_dqn import OnlineMCTSDQNPlayer
from experimental.machine_learning.features import (
    create_sample,
    get_feature_ordering,
)
from experimental.machine_learning.board_tensor_features import (
    CHANNELS,
    HEIGHT,
    WIDTH,
    create_board_tensor,
)
from experimental.machine_learning.utils import (
    get_discounted_return,
    get_tournament_return,
    get_victory_points_return,
    populate_matrices,
    DISCOUNT_FACTOR,
)


RUNNING_AVG_LENGTH = 1


@click.command()
@click.option("-n", "--num", default=5, help="Number of games to play.")
@click.option(
    "--players",
    default="R,R,R,R",
    help="""
        Comma-separated 4 players to use. R=Random, W=WeightedRandom, VX=VRLPlayer(Version X),
            PX=PRLPlayer(Version X), QX=QRLPlayer(Version X). X >= 1.
    """,
)
@click.option(
    "-o",
    "--outpath",
    default=None,
    help="Path where to save ML csvs.",
)
@click.option(
    "--watch/--no-watch",
    default=False,
    help="""
        Whether to save intermediate states to database to allow for viewing.
        This will artificially slow down the game by 1s per move.
        """,
)
def simulate(num, players, outpath, watch):
    """Simple program simulates NUM Catan games."""
    player_keys = players.split(",")

    initialized_players = []
    colors = [c for c in Color]
    pseudonyms = ["Foo", "Bar", "Baz", "Qux"]
    for i, key in enumerate(player_keys):
        player_type = key[0]
        param = key[1:]
        if player_type == "R":
            initialized_players.append(RandomPlayer(colors[i], pseudonyms[i]))
        elif player_type == "H":
            initialized_players.append(HumanPlayer(colors[i], pseudonyms[i]))
        elif player_type == "W":
            initialized_players.append(WeightedRandomPlayer(colors[i], pseudonyms[i]))
        elif player_type == "V":
            initialized_players.append(VRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "Q":
            initialized_players.append(QRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "P":
            initialized_players.append(PRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "T":
            initialized_players.append(TensorRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "M":
            initialized_players.append(MCTSPlayer(colors[i], pseudonyms[i], int(param)))
        elif player_type == "O":
            initialized_players.append(OnlineMCTSDQNPlayer(colors[i], pseudonyms[i]))
        elif player_type == "X":
            initialized_players.append(MiniMaxPlayer(colors[i], pseudonyms[i]))
        elif player_type == "F":
            initialized_players.append(ValueFunctionPlayer(colors[i], pseudonyms[i]))
        elif player_type == "S":
            initialized_players.append(ScikitPlayer(colors[i], pseudonyms[i]))
        else:
            raise ValueError("Invalid player key")

    play_batch(num, initialized_players, outpath, watch)


def play_batch(num_games, players, games_directory, watch):
    """Plays num_games, saves final game in database, and populates data/ matrices"""
    wins = defaultdict(int)
    turns = []
    ticks = []
    durations = []
    games = []
    results_by_player = defaultdict(list)
    writer = tf.summary.create_file_writer(f"logs/play/{int(time.time())}")
    for i in range(num_games):
        for player in players:
            player.restart_state()
        game = Game(players)

        print(f"Playing game {i + 1} / {num_games}. Seating:", game.state.players)
        action_callbacks = []
        if games_directory:
            action_callbacks.append(build_action_callback(games_directory))
        if watch:
            save_game_state(game)

            def callback(game):
                save_game_state(game)
                time.sleep(0.25)

            action_callbacks.append(callback)
            print("Watch game by refreshing http://localhost:3000/games/" + game.id)

        start = time.time()
        try:
            game.play(action_callbacks)
        except Exception as e:
            traceback.print_exc()
        finally:
            duration = time.time() - start
        print("Took", duration, "seconds")
        print({str(p): p.actual_victory_points for p in players})
        if not watch:
            save_game_state(game)
            print("Saved in db. See result at http://localhost:3000/games/" + game.id)
        print("")

        winner = game.winning_player()
        wins[str(winner)] += 1
        turns.append(game.num_turns)
        ticks.append(len(game.state.actions))
        durations.append(duration)
        games.append(game)
        for player in players:
            results_by_player[player.color].append(player.actual_victory_points)
        with writer.as_default():
            for player in players:
                results = results_by_player[player.color]
                last_results = results[len(results) - RUNNING_AVG_LENGTH :]
                if len(last_results) >= RUNNING_AVG_LENGTH:
                    running_avg = sum(last_results) / len(last_results)
                    tf.summary.scalar(
                        f"{player.color}-vp-running-avg", running_avg, step=i
                    )
            writer.flush()

    print("AVG Ticks:", sum(ticks) / len(ticks))
    print("AVG Turns:", sum(turns) / len(turns))
    print("AVG Duration:", sum(durations) / len(durations))

    for player in players:
        vps = results_by_player[player.color]
        print("AVG VPS:", player, sum(vps) / len(vps))

    # Print Winners graph in command line:
    fig = tpl.figure()
    fig.barh([wins[str(p)] for p in players], players, force_ascii=False)
    fig.show()

    return games


def build_action_callback(games_directory):
    data = defaultdict(
        lambda: {
            "samples": [],
            "actions": [],
            "board_tensors": [],
            # These are for practicing ML with simpler problems
            "OWS_ONLY_LABEL": [],
            "OWS_LABEL": [],
            "settlements": [],
            "cities": [],
            "prod_vps": [],
        }
    )

    def action_callback(game: Game):
        if len(game.state.actions) == 0:
            return

        if game.winning_player() is not None:
            flush_to_matrices(game, data, games_directory)
            return

        action = game.state.actions[-1]
        player = game.players_by_color[action.color]
        data[player.color]["samples"].append(create_sample(game, player.color))
        data[player.color]["actions"].append(hot_one_encode_action(action))

        flattened_tensor = tf.reshape(
            create_board_tensor(game, player.color), (WIDTH * HEIGHT * CHANNELS,)
        ).numpy()
        data[player.color]["board_tensors"].append(flattened_tensor)

        player_tiles = set()
        for node_id in (
            player.buildings[BuildingType.SETTLEMENT]
            + player.buildings[BuildingType.CITY]
        ):
            for tile in game.board.get_adjacent_tiles(node_id):
                player_tiles.add(tile.resource)
        data[player.color]["OWS_ONLY_LABEL"].append(
            player_tiles == set([Resource.ORE, Resource.WHEAT, Resource.SHEEP])
        )
        data[player.color]["OWS_LABEL"].append(
            Resource.ORE in player_tiles
            and Resource.WHEAT in player_tiles
            and Resource.SHEEP in player_tiles
        )
        data[player.color]["settlements"].append(
            len(player.buildings[BuildingType.SETTLEMENT])
        )
        data[player.color]["cities"].append(len(player.buildings[BuildingType.CITY]))
        data[player.color]["prod_vps"].append(
            len(player.buildings[BuildingType.SETTLEMENT])
            + len(player.buildings[BuildingType.CITY])
        )

    return action_callback


def flush_to_matrices(game, data, games_directory):
    print("Flushing to matrices...")
    t1 = time.time()
    samples = []
    actions = []
    board_tensors = []
    labels = []
    for player in game.state.players:
        player_data = data[player.color]
        samples.extend(player_data["samples"])
        actions.extend(player_data["actions"])
        board_tensors.extend(player_data["board_tensors"])

        # Make matrix of (RETURN, DISCOUNTED_RETURN, TOURNAMENT_RETURN, DISCOUNTED_TOURNAMENT_RETURN)
        episode_return = get_discounted_return(game, player, 1)
        discounted_return = get_discounted_return(game, player, DISCOUNT_FACTOR)
        tournament_return = get_tournament_return(game, player, 1)
        vp_return = get_victory_points_return(game, player)
        discounted_tournament_return = get_tournament_return(
            game, player, DISCOUNT_FACTOR
        )
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
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["OWS_ONLY_LABEL"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["OWS_LABEL"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["settlements"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["cities"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["prod_vps"]])), axis=1
        )
        labels.extend(return_matrix)

    # Build Q-learning Design Matrix
    samples_df = pd.DataFrame.from_records(
        samples, columns=get_feature_ordering()  # this must be in sync with features.
    ).astype("float64")
    board_tensors_df = pd.DataFrame(board_tensors).astype("float64")
    actions_df = pd.DataFrame(actions).astype("float64").add_prefix("ACTION_")
    rewards_df = pd.DataFrame(
        labels,
        columns=[
            "RETURN",
            "DISCOUNTED_RETURN",
            "TOURNAMENT_RETURN",
            "DISCOUNTED_TOURNAMENT_RETURN",
            "VICTORY_POINTS_RETURN",
            "OWS_ONLY_LABEL",
            "OWS_LABEL",
            "settlements",
            "cities",
            "prod_vps",
        ],
    ).astype("float64")
    print(rewards_df.describe())

    print(
        "Collected DataFrames. Data size:",
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
        samples_df, board_tensors_df, actions_df, rewards_df, games_directory
    )
    print("Saved to matrices at:", games_directory, ". Took", time.time() - t1)
    return samples_df, board_tensors_df, actions_df, rewards_df


if __name__ == "__main__":
    simulate()
