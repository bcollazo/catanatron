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
    second_observation, reward, terminated, truncated, info = env.step(action)
    assert (first_observation != second_observation).any()
    assert reward == 0
    assert not terminated
    assert not truncated
    assert len(env.get_valid_actions()) in [2, 3]

    assert second_observation[features.index("BANK_DEV_CARDS")] == 25
    assert second_observation[features.index("BANK_SHEEP")] == 19
    assert get_p0_num_settlements(second_observation) == 1

    reset_obs, _ = env.reset()
    assert (reset_obs != second_observation).any()
    assert get_p0_num_settlements(reset_obs) == 0

    env.close()


def test_gym_registration_and_api_works():
    env = gym.make("catanatron_gym:catanatron-v1")
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
    env = gym.make(
        "catanatron_gym:catanatron-v1", config={"invalid_action_reward": -1234}
    )
    first_obs, _ = env.reset()
    invalid_action = next(filter(lambda i: i not in env.get_valid_actions(), range(1000)))  # type: ignore
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

    env = gym.make(
        "catanatron_gym:catanatron-v1", config={"reward_function": custom_reward}
    )
    observation, info = env.reset()
    action = random.choice(env.get_valid_actions())  # type: ignore
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 123


def test_custom_map():
    env = gym.make("catanatron_gym:catanatron-v1", config={"map_type": "MINI"})
    observation, info = env.reset()
    assert len(env.get_valid_actions()) < 50  # type: ignore
    assert len(observation) < 614
    # assert env.action_space.n == 260


def test_enemies():
    env = gym.make(
        "catanatron_gym:catanatron-v1",
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
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Virtually impossible for a Random bot to beat Value Function Player
    assert env.game.winning_color() == Color.RED  # type: ignore
    assert reward - 1
    env.close()


def test_mixed_rep():
    env = gym.make(
        "catanatron_gym:catanatron-v1",
        config={"representation": "mixed"},
    )
    observation, info = env.reset()
    assert "board" in observation
    assert "numeric" in observation


def test_resetting_game_reorders_players():
    """In this particular seed sequence, two games is enough to see both orderings"""
    env = CatanatronEnv()
    env.reset(seed=42)
    assert env.game.state.colors == (Color.BLUE, Color.RED)
    env.reset()
    assert env.game.state.colors == (Color.RED, Color.BLUE)


from catanatron.state_functions import get_enemy_colors
from catan_rayrl.catanmultiagentenv import CatanatronMultiAgentEnv


def test_multiagent_env():
    env = CatanatronMultiAgentEnv()
    obs, infos = env.reset(seed=42, options={})
    assert isinstance(obs, dict)
    assert len(obs) == 1

    (color, _) = obs.popitem()
    action = random.choice(env.get_valid_actions())
    enemy_colors = [c.value for c in env.game.state.colors if c.value != color]
    assert len(enemy_colors) == 1
    enemy_color = enemy_colors.pop()

    # build first house
    obs, rewards, terminateds, truncateds, infos = env.step(action_dict={color: action})
    assert rewards == {color: 0}
    # game should not terminate/truncate in first couple of turns
    assert terminateds == {color: False, "__all__": False}
    assert truncateds == {color: False}
    # this dict contains all players
    assert color in infos
    assert "valid_actions" in infos[color]
    assert enemy_color in infos

    # playing invalid action should also have same structure
    invalid_action = None
    while invalid_action is None:
        tmp = env.action_space.sample()
        if tmp not in infos[color]["valid_actions"]:
            invalid_action = tmp
    obs, rewards, terminateds, truncateds, infos = env.step(
        action_dict={color: invalid_action}
    )
    assert rewards == {color: -1}
    # game should not terminate/truncate in an invalid action
    assert terminateds == {color: False, "__all__": False}
    assert truncateds == {color: False}
    assert color in infos
    assert "valid_actions" in infos[color]
    assert enemy_color in infos

    # next turn should be placing roads
