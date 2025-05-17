import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from catanatron.features import (
    create_sample,
    create_sample_vector,
    get_feature_ordering,
)
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY, ACTION_SPACE_SIZE
from catanatron.gym.board_tensor_features import (
    NUMERIC_FEATURES,
    create_board_tensor,
)

# from catanatron_experimental.rep_b_model import build_model

# Taken from correlation analysis
FEATURES = [
    "P0_HAS_ROAD",
    "P1_HAS_ROAD",
    "P0_HAS_ARMY",
    "P1_HAS_ARMY",
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
    "P0_PUBLIC_VPS",
    "P1_PUBLIC_VPS",
    "P0_SETTLEMENTS_LEFT",
    "P1_SETTLEMENTS_LEFT",
    "P0_CITIES_LEFT",
    "P1_CITIES_LEFT",
    "P0_KNIGHT_PLAYED",
    "P1_KNIGHT_PLAYED",
]


def allow_feature(feature_name):
    return (
        "2_ROAD" not in feature_name
        and "HAND" not in feature_name
        and "BANK" not in feature_name
        and "P0_ACTUAL_VPS" != feature_name
        and "PLAYABLE" not in feature_name
        and "LEFT" not in feature_name
        and "ROLLED" not in feature_name
        and "PLAYED" not in feature_name
        and "PUBLIC_VPS" not in feature_name
        and not ("TOTAL" in feature_name and "P1" in feature_name)
        and not ("EFFECTIVE" in feature_name and "P0" in feature_name)
        and (feature_name[-6:] != "PLAYED" or "KNIGHT" in feature_name)
    )


ALL_FEATURES = get_feature_ordering(num_players=2)
FEATURES = list(filter(allow_feature, ALL_FEATURES))
FEATURES = get_feature_ordering(2)
FEATURE_INDICES = [ALL_FEATURES.index(f) for f in FEATURES]

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
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)

    return normalized


class PRLPlayer(Player):
    def __init__(self, color, model_path):
        super(PRLPlayer, self).__init__(color)
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
    def __init__(self, color, model_path):
        super(QRLPlayer, self).__init__(color)
        self.model_path = model_path
        global Q_MODEL
        Q_MODEL = keras.models.load_model(model_path)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # epsilon-greedy: with EPSILON probability play at random.
        # if random.random() < EPSILON:
        #     index = random.randrange(0, len(playable_actions))
        #     return playable_actions[index]

        # Create sample matrix of state + action vectors.
        state = create_sample_vector(game, self.color, FEATURES)
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
    def __init__(self, color, model_path):
        super(VRLPlayer, self).__init__(color)
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

        # We do this instead of np.argmax(scores), because often all have same
        #   value, at which point we want random instead of first (end turn).
        best_score = np.max(scores)
        max_indices = np.where(scores == best_score)
        best_idx = np.random.choice(max_indices[0])

        # pprint(list(zip(FEATURES, get_v_model(self.model_path).get_weights()[-2])))
        # pprint(list(zip(playable_actions, scores)))
        # breakpoint()
        return playable_actions[best_idx]

    def __repr__(self):
        return super(VRLPlayer, self).__repr__() + f"({self.model_path})"


class TensorRLPlayer(Player):
    def __init__(self, color, model_path):
        super(TensorRLPlayer, self).__init__(color)
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
