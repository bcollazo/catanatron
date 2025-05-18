import random

import gymnasium
from gymnasium.utils.env_checker import check_env
import numpy as np

from catanatron.features import get_feature_ordering
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import CatanatronEnv

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
    def custom_reward(game, p0_color):
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
