import random

import gymnasium
from gymnasium.utils.env_checker import check_env
import numpy as np

from catanatron.features import get_feature_ordering
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import ActionType
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.gym.envs.action_space import get_action_array

features = get_feature_ordering(2)


def get_p0_num_settlements(obs):
    indexes = [
        i
        for i, name in enumerate(features)
        if "NODE" in name and "SETTLEMENT" in name and "P0" in name
    ]
    return sum([obs[i] for i in indexes])


def test_check_env():
    env = CatanatronEnv()
    check_env(env)


def test_gym():
    env = CatanatronEnv()

    first_observation, info = env.reset()  # this forces advanced until p0...
    assert len(info["valid_actions"]) >= 50  # first seat at most blocked 4 nodes
    assert get_p0_num_settlements(first_observation) == 0

    action = random.choice(info["valid_actions"])
    second_observation, reward, terminated, truncated, info = env.step(action)
    assert np.any(first_observation != second_observation)
    assert reward == 0
    assert not terminated
    assert not truncated
    assert len(info["valid_actions"]) in [2, 3]

    assert second_observation[features.index("BANK_DEV_CARDS")] == 25  # type: ignore
    assert second_observation[features.index("BANK_SHEEP")] == 19  # type: ignore
    assert get_p0_num_settlements(second_observation) == 1

    reset_obs, _ = env.reset()
    assert np.any(reset_obs != second_observation)
    assert get_p0_num_settlements(reset_obs) == 0

    env.close()


def test_gym_registration_and_api_works():
    env = gymnasium.make("catanatron/Catanatron-v0")
    observation, info = env.reset()
    done = False
    reward = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
    assert reward in [-1, 1]


def test_invalid_action_reward():
    env = gymnasium.make(
        "catanatron/Catanatron-v0", config={"invalid_action_reward": -1234}
    )
    first_obs, info = env.reset()
    invalid_action = next(filter(lambda i: i not in info["valid_actions"], range(1000)))
    observation, reward, terminated, truncated, info = env.step(invalid_action)
    assert reward == -1234
    assert not terminated
    assert not truncated
    assert (observation == first_obs).all()
    for _ in range(500):
        observation, reward, terminated, truncated, info = env.step(invalid_action)
        assert (observation == first_obs).all()
    assert not terminated
    assert truncated


def test_custom_reward():
    def custom_reward(action, game, p0_color):
        return 123

    env = gymnasium.make(
        "catanatron/Catanatron-v0", config={"reward_function": custom_reward}
    )
    observation, info = env.reset()
    action = random.choice(info["valid_actions"])
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 123


def test_custom_map():
    env = gymnasium.make("catanatron/Catanatron-v0", config={"map_type": "MINI"})
    observation, info = env.reset()
    assert len(info["valid_actions"]) < 50
    assert len(observation) < 614
    # assert env.action_space.n == 260


def test_enemies():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [
                ValueFunctionPlayer(Color.RED),
                RandomPlayer(Color.ORANGE),
                RandomPlayer(Color.WHITE),
            ]
        },
    )
    observation, info = env.reset()
    assert len(observation) == len(get_feature_ordering(4))

    done = False
    reward = 0
    while not done:
        action = random.choice(info["valid_actions"])
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Virtually impossible for a Random bot to beat Value Function Player
    assert env.unwrapped.game.winning_color() == Color.RED  # type: ignore
    assert reward == -1  # ensure we lost
    env.close()


def test_mixed_rep():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={"representation": "mixed"},
    )
    observation, info = env.reset()
    assert "board" in observation
    assert "numeric" in observation


def test_move_robber_action_in_base_action_array():
    """Test that a specific MOVE_ROBBER action is in the BASE action array for 2 players."""
    player_colors = (Color.BLUE, Color.RED)
    action_array = get_action_array(player_colors, "BASE")
    target_action = (ActionType.MOVE_ROBBER, ((-1, 0, 1), Color.BLUE))
    assert target_action in action_array, (
        f"Action {target_action} not found in BASE action array for 2 players"
    )

    target_action = (ActionType.MOVE_ROBBER, ((-1, 0, 1), None))
    assert target_action in action_array, (
        f"Action {target_action} not found in BASE action array for 2 players"
    )


def test_there_are_54_build_nodes_in_base():
    player_colors = (Color.BLUE, Color.RED)
    action_array = get_action_array(player_colors, "BASE")
    num_build_nodes = len(
        [action for action in action_array if action[0] == ActionType.BUILD_SETTLEMENT]
    )
    assert num_build_nodes == 54


def test_there_are_less_build_nodes_in_mini():
    player_colors = (Color.BLUE, Color.RED)
    action_array = get_action_array(player_colors, "MINI")
    num_build_nodes = len(
        [action for action in action_array if action[0] == ActionType.BUILD_SETTLEMENT]
    )
    assert num_build_nodes == 24


def test_outside_tiles_not_in_mini():
    player_colors = (Color.BLUE, Color.RED)
    action_array = get_action_array(player_colors, "MINI")
    target_action = (ActionType.MOVE_ROBBER, ((0, 2, -2), Color.BLUE))
    assert target_action not in action_array, (
        f"Action {target_action} found in MINI action array for 2 players"
    )
