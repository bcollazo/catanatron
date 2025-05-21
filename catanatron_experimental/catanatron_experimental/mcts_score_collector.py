import os
from typing import Iterable
from catanatron.utils import ensure_dir

import pandas as pd
import tensorflow as tf

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.enums import SETTLEMENT, CITY
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.playouts import run_playouts
from catanatron.features import (
    build_production_features,
    resource_hand_features,
)


def simple_feature_vector(game, p0_color):
    key = player_key(game.state, p0_color)
    f_public_vps = game.state.player_state[f"P0_VICTORY_POINTS"]
    f_enemy_public_vps = game.state.player_state[f"P1_VICTORY_POINTS"]

    production_features = build_production_features(True)
    our_production_sample = production_features(game, p0_color)
    enemy_production_sample = production_features(game, p0_color)

    f_longest_road_length = game.state.player_state["P0_LONGEST_ROAD_LENGTH"]
    f_enemy_longest_road_length = game.state.player_state["P1_LONGEST_ROAD_LENGTH"]

    hand_sample = resource_hand_features(game, p0_color)
    distance_to_city = (
        max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
        + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
    ) / 5.0  # 0 means good. 1 means bad.
    distance_to_settlement = (
        max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
        + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
        + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
        + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
    ) / 4.0  # 0 means good. 1 means bad.
    f_hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

    f_num_in_hand = player_num_resource_cards(game.state, p0_color)

    # blockability
    buildings = game.state.buildings_by_color[p0_color]
    owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
    owned_tiles = set()
    for n in owned_nodes:
        owned_tiles.update(game.state.board.map.adjacent_tiles[n])
    # f_num_tiles = len(owned_tiles)

    # f_num_buildable_nodes = len(game.state.board.buildable_node_ids(p0_color))
    # f_hand_devs = player_num_dev_cards(game.state, p0_color)

    f_army_size = game.state.player_state[f"P1_PLAYED_KNIGHT"]
    f_enemy_army_size = game.state.player_state[f"P0_PLAYED_KNIGHT"]

    vector = {
        # Where to place. Note winning is best at all costs
        "PUBLIC_VPS": f_public_vps,
        "ENEMY_PUBLIC_VPS": f_enemy_public_vps,
        # "NUM_TILES": f_num_tiles,
        # "BUILDABLE_NODES": f_num_buildable_nodes,
        # Hand, when to hold and when to use.
        "HAND_SYNERGY": f_hand_synergy,
        "HAND_RESOURCES": f_num_in_hand,
        # "HAND_DEVS": f_hand_devs,
        # Other
        "ROAD_LENGTH": f_longest_road_length,
        "ENEMY_ROAD_LENGTH": f_enemy_longest_road_length,
        "ARMY_SIZE": f_army_size,
        "ENEMY_ARMY_SIZE": f_enemy_army_size,
    }
    vector = {**vector, **our_production_sample}
    vector = {**vector, **enemy_production_sample}
    return vector


NUM_SIMULATIONS = 100
RECORDS = []
DATA_DIRECTORY = "data/mcts-collector"
DATASET_PATH = os.path.join(DATA_DIRECTORY, "simple.csv.gz")


class MCTSScoreCollector(AlphaBetaPlayer):
    def reset_state(self):
        global RECORDS
        super().reset_state()

        if len(RECORDS) > 0:  # Flush data to disk
            ensure_dir(DATA_DIRECTORY)
            is_first_training = not os.path.isfile(DATASET_PATH)
            df = pd.DataFrame.from_records(RECORDS).astype("float64")
            df.to_csv(
                DATASET_PATH,
                mode="a",
                header=is_first_training,
                index=False,
                compression="gzip",
            )
            RECORDS = []
            print("Flushed dataset of shape:", df.shape)

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        decided_action = super().decide(game, playable_actions)

        # Log simple dataset of simple features and MCTS Score
        results = run_playouts(game.copy(), NUM_SIMULATIONS)
        vector = simple_feature_vector(game, self.color)
        vector["LABEL"] = results[self.color] / float(NUM_SIMULATIONS)
        RECORDS.append(vector)

        return decided_action


# Singleton-pattern
MCTS_PREDICTOR_MODEL = None


class MCTSPredictor(AlphaBetaPlayer):
    def __init__(self, *params, **kwargs):
        super().__init__(*params, **kwargs)
        self.use_value_function = True

    def value_function(self, game, p0_color):
        global MCTS_PREDICTOR_MODEL
        if MCTS_PREDICTOR_MODEL is None:
            MCTS_PREDICTOR_MODEL = tf.keras.models.load_model(
                "catanatron_experimental/catanatron_experimental/notebooks/models/simple-mcts-score-predictor"
            )
        model = MCTS_PREDICTOR_MODEL

        vector = simple_feature_vector(game, p0_color)
        df = pd.DataFrame.from_records([vector])
        score = model.call(df)[0][0].numpy()

        return score

    def decide(self, game: Game, playable_actions):
        decided_action = super().decide(game, playable_actions)

        return decided_action
