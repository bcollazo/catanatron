from pathlib import Path

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorforce import Agent, Environment
import click

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from experimental.machine_learning.features import (
    create_sample_vector,
    get_feature_ordering,
)
from experimental.machine_learning.players.reinforcement import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    normalize_action,
)


FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)
EPISODES = 25000


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
    if checkpoint_directory.exists():
        print("Loading model...")
        agent = Agent.load(directory=str(checkpoint_directory))
    else:
        print("Creating model...")
        agent = Agent.create(
            agent="tensorforce",
            environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
            memory=50000,  # alphazero is 500,000
            update=dict(unit="episodes", batch_size=32),
            optimizer=dict(type="adam", learning_rate=10e-3),
            policy=dict(network="auto"),
            exploration=0.10,
            # policy=dict(network=dict(type='layered', layers=[dict(type='dense', size=32)])),
            objective="policy_gradient",
            reward_estimation=dict(horizon=20),
            l2_regularization=10e-2,
            summarizer=dict(
                directory="data/logs",
                summaries=["reward", "action-value"],
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
        reward = (
            int(winning_color == self.p0.color) * 1000 + self.p0.actual_victory_points
        )
        return next_state, terminal, reward

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_player() is None
            and self.game.state.current_player().color != self.p0.color
        ):
            self.game.play_tick()  # will play bot


def build_states(game, p0):
    sample = create_sample_vector(game, p0.color)

    action_ints = list(map(to_action_space, game.state.playable_actions))
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[action_ints] = True

    states = dict(
        state=sample,
        action_mask=mask,
    )
    return states


MODEL = None


class ForcePlayer(Player):
    def __init__(self, color, name):
        super(ForcePlayer, self).__init__(color, name)
        global MODEL
        MODEL = Agent.load(directory="data/checkpoints")

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        states = build_states(game, self)
        action_int = MODEL.act(states, independent=True)
        best_action = from_action_space(action_int, playable_actions)
        return best_action


def create_model():
    inputs = tf.keras.Input(shape=(NUM_FEATURES,))
    outputs = tf.keras.layers.Dense(32, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(ACTION_SPACE_SIZE)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
    return model


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
