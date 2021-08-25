import random
from pathlib import Path
import logging

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorforce import Agent, Environment
import click

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    normalize_action,
)

# For repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)
EPISODES = 35_000  # 25_000 is like 8 hours


@click.command()
@click.argument("experiment_name")
def main(experiment_name):
    # Pre-defined or custom environment
    # environment = Environment.create(
    #     environment="gym", level="CartPole", max_episode_timesteps=500
    # )
    environment = Environment.create(
        environment=CustomEnvironment, max_episode_timesteps=1000
    )

    checkpoint_directory = Path("data/checkpoints/", experiment_name)
    logs_directory = Path("data/logs", experiment_name)
    if checkpoint_directory.exists():
        print("Loading model...")
        agent = Agent.load(directory=str(checkpoint_directory))
    else:
        print("Creating model...")
        agent = Agent.create(
            agent="vpg",
            environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
            memory=50_000,  # alphazero is 500,000
            batch_size=32,
            update_frequency=1,
            # update=dict(unit="episodes", batch_size=32),
            # optimizer=dict(type="adam", learning_rate=1e-3),
            # policy=dict(network="auto"),
            # exploration=0.05,
            exploration=dict(
                type="linear",
                unit="episodes",
                num_steps=EPISODES,
                initial_value=1.0,
                final_value=0.05,
            ),
            # policy=dict(network=dict(type='layered', layers=[dict(type='dense', size=32)])),
            # objective="policy_gradient",
            # reward_estimation=dict(horizon=20, discount=0.999),
            l2_regularization=1e-4,
            summarizer=dict(
                directory=str(logs_directory),
                summaries=["reward", "action-value", "parameters"],
            ),
            saver=dict(
                directory=str(checkpoint_directory),
                frequency=100,  # save checkpoint every 100 updates
            ),
        )

    # Train for 300 episodes
    for _ in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):

        # Initialize episode
        states = environment.reset()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

    agent.close()
    environment.close()


class CustomEnvironment(Environment):
    def states(self):
        return dict(type="float", shape=(NUM_FEATURES,))

    def actions(self):
        return dict(type="int", num_values=ACTION_SPACE_SIZE)

    def reset(self):
        p0 = Player(Color.BLUE)
        players = [
            p0,
            RandomPlayer(Color.RED),
            # VictoryPointPlayer(Color.RED),
        ]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0_decision()
        return build_states(self.game, self.p0)

    def execute(self, actions):
        action = from_action_space(actions, self.game.state.playable_actions)
        self.game.execute(action)
        self._advance_until_p0_decision()

        winning_color = self.game.winning_color()
        next_state = build_states(self.game, self.p0)
        terminal = winning_color is not None

        # key = player_key(self.game.state, self.p0.color)
        # points = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        # reward = int(winning_color == self.p0.color) * 1000 + points
        if self.p0.color == winning_color:
            reward = 1
        elif winning_color is None:
            reward = 0
        else:
            reward = -1
        return next_state, terminal, reward

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_player().color != self.p0.color
        ):
            self.game.play_tick()  # will play bot


def build_states(game, p0):
    sample = create_sample_vector(game, p0.color)

    action_ints = list(map(to_action_space, game.state.playable_actions))
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[action_ints] = True

    states = dict(state=sample, action_mask=mask)
    return states


MODEL = None


class ForcePlayer(Player):
    def __init__(self, color, model_name):
        super(ForcePlayer, self).__init__(color)
        global MODEL
        MODEL = Agent.load(directory="data/checkpoints/" + model_name)
        MODEL.spec["summarizer"] = None
        MODEL.spec["saver"] = None
        logging.getLogger().handlers = []

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        states = build_states(game, self)
        action_int = MODEL.act(states, independent=True)
        best_action = from_action_space(action_int, playable_actions)
        return best_action


def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint
    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action


if __name__ == "__main__":
    main()
