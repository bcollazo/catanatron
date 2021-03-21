import time
import random
import os
import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.actions import Action, ActionType, TradeOffer
from catanatron.models.map import BaseMap, NUM_NODES, Tile
from catanatron.models.board import get_edges
from catanatron.models.enums import Resource
from experimental.machine_learning.features import (
    create_sample,
    create_sample_vector,
)
from experimental.machine_learning.board_tensor_features import (
    NUMERIC_FEATURES,
    create_board_tensor,
)

# from experimental.rep_b_model import build_model

# Taken from correlation analysis
FEATURES = [
    "P0_HAS_ROAD",
    "P1_HAS_ROAD",
    "P2_HAS_ROAD",
    "P3_HAS_ROAD",
    "P0_HAS_ARMY",
    "P1_HAS_ARMY",
    "P2_HAS_ARMY",
    "P3_HAS_ARMY",
    "P0_ORE_PRODUCTION",
    "P0_WOOD_PRODUCTION",
    "P0_WHEAT_PRODUCTION",
    "P0_SHEEP_PRODUCTION",
    "P0_BRICK_PRODUCTION",
    "P0_LONGEST_ROAD_LENGTH",
    "P1_ORE_PRODUCTION",
    "P1_WOOD_PRODUCTION",
    "P1_WHEAT_PRODUCTION",
    "P1_SHEEP_PRODUCTION",
    "P1_BRICK_PRODUCTION",
    "P1_LONGEST_ROAD_LENGTH",
    "P2_ORE_PRODUCTION",
    "P2_WOOD_PRODUCTION",
    "P2_WHEAT_PRODUCTION",
    "P2_SHEEP_PRODUCTION",
    "P2_BRICK_PRODUCTION",
    "P2_LONGEST_ROAD_LENGTH",
    "P3_ORE_PRODUCTION",
    "P3_WOOD_PRODUCTION",
    "P3_WHEAT_PRODUCTION",
    "P3_SHEEP_PRODUCTION",
    "P3_BRICK_PRODUCTION",
    "P3_LONGEST_ROAD_LENGTH",
    "P0_PUBLIC_VPS",
    "P1_PUBLIC_VPS",
    "P2_PUBLIC_VPS",
    "P3_PUBLIC_VPS",
    "P0_SETTLEMENTS_LEFT",
    "P1_SETTLEMENTS_LEFT",
    "P2_SETTLEMENTS_LEFT",
    "P3_SETTLEMENTS_LEFT",
    "P0_CITIES_LEFT",
    "P1_CITIES_LEFT",
    "P2_CITIES_LEFT",
    "P3_CITIES_LEFT",
    "P0_KNIGHT_PLAYED",
    "P1_KNIGHT_PLAYED",
    "P2_KNIGHT_PLAYED",
    "P3_KNIGHT_PLAYED",
]


BASE_TOPOLOGY = BaseMap().topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == Tile]
RESOURCE_LIST = list(Resource)
ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    (ActionType.MOVE_ROBBER, None),
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_FIRST_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_INITIAL_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SECOND_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    # TODO: Should we use a heuristic for this?
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCE_LIST[j]))
        for i, first_card in enumerate(RESOURCE_LIST)
        for j in range(i, len(RESOURCE_LIST))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCE_LIST],
    # TODO: consider simetric options to reduce complexity by half.
    *[
        (ActionType.PLAY_ROAD_BUILDING, (tuple(sorted(i)), tuple(sorted(j))))
        for i in get_edges()
        for j in get_edges()
        if i != j
    ],
    *[(ActionType.PLAY_MONOPOLY, r) for r in Resource],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, TradeOffer(tuple(4 * [i]), tuple([j]), None))
        for i in Resource
        for j in Resource
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, TradeOffer(tuple(3 * [i]), tuple([j]), None))
        for i in Resource
        for j in Resource
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, TradeOffer(tuple(2 * [i]), tuple([j]), None))
        for i in Resource
        for j in Resource
        if i != j
    ],
    (ActionType.END_TURN, None),
]

ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)
EPSILON = 0.20  # for epsilon-greedy action selection
# singleton for model. lazy-initialize to easy dependency graph and stored
#   here instead of class attribute to skip saving model in CatanatronDB.
P_MODEL = None
Q_MODEL = None
V_MODEL = None
T_MODEL = None


def v_model_path(version):
    return os.path.join(os.path.dirname(__file__), "v_models", str(version))


def q_model_path(version):
    return os.path.join(os.path.dirname(__file__), "q_models", str(version))


def p_model_path(version):
    return os.path.join(os.path.dirname(__file__), "p_models", str(version))


def get_v_model(model_path):
    global V_MODEL
    if V_MODEL is None:
        custom_objects = None if model_path[:2] != "ak" else ak.CUSTOM_OBJECTS
        V_MODEL = keras.models.load_model(model_path, custom_objects=custom_objects)
    return V_MODEL


def get_t_model(model_path):
    global T_MODEL
    if T_MODEL is None:
        T_MODEL = keras.models.load_model(model_path)
        # T_MODEL = build_model()
    return T_MODEL


def hot_one_encode_action(action):
    normalized = normalize_action(action)
    index = ACTIONS_ARRAY.index((normalized.action_type, normalized.value))
    vector = np.zeros(ACTION_SPACE_SIZE, dtype=int)
    vector[index] = 1
    return vector


def normalize_action(action):
    normalized = copy.deepcopy(action)
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif (
        normalized.action_type == ActionType.MOVE_ROBBER
        or normalized.action_type == ActionType.PLAY_KNIGHT_CARD
    ):
        return Action(action.color, action.action_type, None)
    elif (
        normalized.action_type == ActionType.BUILD_INITIAL_ROAD
        or normalized.action_type == ActionType.BUILD_ROAD
    ):
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.PLAY_ROAD_BUILDING:
        i, j = action.value
        return Action(
            action.color, action.action_type, (tuple(sorted(i)), tuple(sorted(j)))
        )
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)

    return normalized


class PRLPlayer(Player):
    def __init__(self, color, name, model_path):
        super(PRLPlayer, self).__init__(color, name)
        global P_MODEL
        P_MODEL = keras.models.load_model(model_path)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # epsilon-greedy: with EPSILON probability play at random.
        # if random.random() < EPSILON:
        #     print("DOING EPSILON GUESS")
        #     index = random.randrange(0, len(playable_actions))
        #     return playable_actions[index]

        # Create array like [0,0,1,0,0,0,1,...] representing possible actions
        normalized_playable = [normalize_action(a) for a in playable_actions]
        possibilities = [(a.action_type, a.value) for a in normalized_playable]
        possible_indices = [ACTIONS_ARRAY.index(x) for x in possibilities]
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
        mask[possible_indices] = 1

        # possibilities = [(a.action_type, a.value) for a in playable_actions]
        # possible_indices = [ACTIONS_ARRAY.index(x) for x in possibilities]
        # mask = np.zeros(ACTION_SPACE_SIZE, dtype=int)
        # mask[possible_indices] = 1

        # Get action probabilities with neural network.
        X = [create_sample_vector(game, self.color, FEATURES)]
        result = P_MODEL.call(tf.convert_to_tensor(X))

        # Multiply mask with output, and take max.
        clipped_probabilities = np.multiply(mask, result[0])
        action_index = np.argmax(clipped_probabilities)
        playable_actions_index = possibilities.index(ACTIONS_ARRAY[action_index])
        return playable_actions[playable_actions_index]


class QRLPlayer(Player):
    def __init__(self, color, name, version=1):
        super(QRLPlayer, self).__init__(color, name)
        self.version = version
        global Q_MODEL
        Q_MODEL = keras.models.load_model(q_model_path(version))

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # epsilon-greedy: with EPSILON probability play at random.
        # if random.random() < EPSILON:
        #     index = random.randrange(0, len(playable_actions))
        #     return playable_actions[index]

        # Create sample matrix of state + action vectors.
        state = create_sample_vector(game, self.color)
        samples = []
        for action in playable_actions:
            samples.append(np.concatenate((state, hot_one_encode_action(action))))
        X = np.array(samples)

        # Predict on all samples
        result = Q_MODEL.predict(X)
        index = np.argmax(result)
        return playable_actions[index]


# Incremental Value Function Approximation
# Play game, take reward G (discounted by num turns), then for each
#   state S (sample): change params by delta_w = alpha * (G - v(S)) grad_w(S)
class VRLPlayer(Player):
    def __init__(self, color, name, model_path):
        super(VRLPlayer, self).__init__(color, name)
        self.model_path = model_path

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # epsilon-greedy: with EPSILON probability play at random.
        # if random.random() < EPSILON:
        #     index = random.randrange(0, len(playable_actions))
        #     return playable_actions[index]

        # Make copy of each action, and take one that takes to most value.
        samples = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            sample = create_sample(game_copy, self.color)
            state = [float(sample[i]) for i in FEATURES]
            samples.append(state)

        scores = get_v_model(self.model_path).call(tf.convert_to_tensor(samples))
        best_idx = np.argmax(scores)
        return playable_actions[best_idx]

    def __repr__(self):
        return super(VRLPlayer, self).__repr__() + f"({self.model_path})"


class TensorRLPlayer(Player):
    def __init__(self, color, name, model_path):
        super(TensorRLPlayer, self).__init__(color, name)
        self.model_path = model_path

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        # epsilon-greedy: with EPSILON probability play at random.
        # if random.random() < EPSILON:
        #     index = random.randrange(0, len(playable_actions))
        #     return playable_actions[index]

        # Make copy of each action, and take one that takes to most value.
        inputs1 = []
        inputs2 = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            board_tensor = create_board_tensor(game_copy, self.color)
            inputs1.append(board_tensor)

            sample = create_sample(game_copy, self.color)
            input2 = [float(sample[i]) for i in NUMERIC_FEATURES]
            inputs2.append(input2)

        scores = get_t_model(self.model_path).call(
            [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
        )
        best_idx = np.argmax(scores)
        # breakpoint()
        return playable_actions[best_idx]

    def __repr__(self):
        return super(TensorRLPlayer, self).__repr__() + f"({self.model_path})"


class MCTSRLPlayer(Player):
    def __init__(self, color, name, model_path):
        super(MCTSRLPlayer, self).__init__(color, name)
        self.model_path = model_path

    def decide(self, game: Game, playable_actions):
        # for each playable_action, play it in a copy,
        # At each node. Take most likely K plays from policy prediction.
        #   Consider play (make game copy and apply).
        #   Repeat until reach N levels deep.
        # Should profile performance of this. AVG time per level. AVG branching factor.
        # Do min-max? Backpropagate values.
        # Keep dictio of level0play => worst_outcome value.
        # Keep dictio of level0play => best_outcome value.
        TICKS_IN_THE_FUTURE = 4
        TICK_DIMENSIONS = 1  # kinda like PLAYOUTS per node.
        simulation_decide = build_simulation_decide(self.model_path)

        time0 = time.time()

        # Initialize search agenda
        top_k_actions = playable_actions[:]  # TODO: select top k
        best_action = None
        best_action_score = None
        for action in top_k_actions:
            time1 = time.time()
            action_copy = copy.deepcopy(action)
            game_copy = game.copy()
            # TODO: Use (probas, outcomes) = game.execute_all(action_copy, decide)
            # TODO: Append all possible outcomes (not selected only). game.execute_all
            transition_proba = game_copy.execute(action_copy)

            total_compounded_probas = 0
            total_score = 0
            for i in range(TICK_DIMENSIONS):
                game_copy_copy = game_copy.copy()
                compounded_proba = transition_proba
                time2 = time.time()
                for j in range(TICKS_IN_THE_FUTURE):
                    proba = game_copy_copy.play_tick(decide_fn=simulation_decide)
                    compounded_proba *= proba
                # print("Time rolling TICKS_IN_THE_FUTURE", time.time() - time2)
                # now we have an advanced leaf with proba "cp". score it.
                time3 = time.time()
                sample = create_sample_vector(game_copy_copy, self.color)
                score = get_v_model(self.model_path).call(
                    tf.convert_to_tensor([sample])
                )[0]
                # print("Time scoring advanced leaf", time.time() - time3)
                total_score = compounded_proba * score
                total_compounded_probas += compounded_proba

            action_advanced_leaf_weighted_score = total_score / total_compounded_probas
            if (
                best_action is None
                or best_action_score < action_advanced_leaf_weighted_score
            ):
                best_action = action
                best_action_score = action_advanced_leaf_weighted_score

            # print("Time exploring an action:", time.time() - time1)

        # print("Time deciding...", time.time() - time0)
        return best_action

        # scores = get_v_model(self.model_path).call(tf.convert_to_tensor(samples))
        # best_idx = np.argmax(scores)
        # return playable_actions[best_idx]

    def __repr__(self):
        return super(MCTSRLPlayer, self).__repr__() + f"({self.model_path})"


# need to overwrite de decide(self, method) to


def build_simulation_decide(model_path):
    def decide(self, game, playable_actions):
        possible_actions = {
            ActionType.ROLL,  # ill know this if you can roll
            ActionType.MOVE_ROBBER,  # we know this 100% since this is via prompt. its play_knight the one we might not know
            ActionType.DISCARD,
            ActionType.BUILD_FIRST_SETTLEMENT,
            ActionType.BUILD_SECOND_SETTLEMENT,
            ActionType.BUILD_INITIAL_ROAD,
            ActionType.END_TURN,
        }
        # TODO: Use a better estimate of hand, instead of num_cards
        num_resource_cards = self.resource_deck.num_cards()
        num_dev_cards = self.development_deck.num_cards()
        if num_resource_cards >= 2:
            possible_actions.add(ActionType.BUILD_ROAD)
        if num_resource_cards >= 3:
            possible_actions.add(ActionType.BUY_DEVELOPMENT_CARD)
        if num_resource_cards >= 4:
            possible_actions.add(ActionType.BUILD_SETTLEMENT)
        if num_dev_cards >= 1:
            possible_actions.add(ActionType.PLAY_KNIGHT_CARD)
            possible_actions.add(ActionType.PLAY_YEAR_OF_PLENTY)
            possible_actions.add(ActionType.PLAY_MONOPOLY)
            possible_actions.add(ActionType.PLAY_ROAD_BUILDING)

        guessable_actions = list(
            filter(lambda a: a.action_type in possible_actions, playable_actions)
        )
        # index = random.randrange(0, len(guessable_actions))
        # return guessable_actions[index]

        samples = []
        for action in guessable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            state = create_sample_vector(game_copy, self.color)
            samples.append(state)

        scores = get_v_model(model_path).call(tf.convert_to_tensor(samples))
        best_idx = np.argmax(scores)
        return playable_actions[best_idx]

    return decide
