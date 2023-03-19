import random

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from catanatron_gym.features import get_feature_ordering
from catanatron.models.player import Color, RandomPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_gym.envs.catanatron_env import CatanatronEnv

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

    first_observation, _ = env.reset()  # this forces advanced until p0...
    assert len(env.get_valid_actions()) >= 50  # first seat at most blocked 4 nodes
    assert get_p0_num_settlements(first_observation) == 0

    action = random.choice(env.get_valid_actions())
    second_observation, reward, done, info = env.step(action)
    assert (first_observation != second_observation).any()
    assert reward == 0
    assert not done
    assert len(env.get_valid_actions()) in [2, 3]

    assert second_observation[features.index("BANK_DEV_CARDS")] == 25
    assert second_observation[features.index("BANK_SHEEP")] == 19
    assert get_p0_num_settlements(second_observation) == 1

    reset_obs, _ = env.reset()
    assert (reset_obs != second_observation).any()
    assert get_p0_num_settlements(reset_obs) == 0

    env.close()


def test_gym_registration_and_api_works():
    env = gym.make("catanatron_gym:catanatron-v0")
    observation, _ = env.reset()
    done = False
    reward = 0
    while not done:
        action = random.choice(env.get_valid_actions())  # type: ignore
        observation, reward, done, info = env.step(action)
    env.close()
    assert reward in [-1, 1]


def test_invalid_action_reward():
    env = gym.make(
        "catanatron_gym:catanatron-v0", config={"invalid_action_reward": -1234}
    )
    first_obs, _ = env.reset()
    invalid_action = next(filter(lambda i: i not in env.get_valid_actions(), range(1000)))  # type: ignore
    observation, reward, done, info = env.step(invalid_action)
    assert reward == -1234
    assert not done
    assert (observation == first_obs).all()
    for _ in range(500):
        observation, reward, done, info = env.step(invalid_action)
        assert (observation == first_obs).all()
    assert done


def test_custom_reward():
    def custom_reward(game, p0_color):
        return 123

    env = gym.make(
        "catanatron_gym:catanatron-v0", config={"reward_function": custom_reward}
    )
    observation, info = env.reset()
    action = random.choice(env.get_valid_actions())  # type: ignore
    observation, reward, done, info = env.step(action)
    assert reward == 123


def test_custom_map():
    env = gym.make("catanatron_gym:catanatron-v0", config={"map_type": "MINI"})
    observation, info = env.reset()
    assert len(env.get_valid_actions()) < 50  # type: ignore
    assert len(observation) < 614
    # assert env.action_space.n == 260


def test_enemies():
    env = gym.make(
        "catanatron_gym:catanatron-v0",
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
        action = random.choice(env.get_valid_actions())  # type: ignore
        observation, reward, done, info = env.step(action)

    # Virtually impossible for a Random bot to beat Value Function Player
    assert env.game.winning_color() == Color.RED  # type: ignore
    assert reward - 1
    env.close()


def test_mixed_rep():
    env = gym.make(
        "catanatron_gym:catanatron-v0",
        config={"representation": "mixed"},
    )
    observation, info = env.reset()
    assert "board" in observation
    assert "numeric" in observation
