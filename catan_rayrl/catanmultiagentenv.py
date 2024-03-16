from typing import Optional
from collections import defaultdict

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.utils import try_import_pyspiel

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)
from catanatron_gym.envs.catanatron_env import (
    CatanatronEnv,
    ACTION_SPACE_SIZE,
    HIGH,
    NUM_FEATURES,
    simple_reward,
    to_action_space,
    from_action_space,
)
