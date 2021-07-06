import os
import time
import random
import sys, traceback
from pathlib import Path
import click
from collections import Counter, deque

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tqdm import tqdm
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from catanatron.state import player_key
from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from experimental.machine_learning.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_server.utils import ensure_link
from experimental.machine_learning.board_tensor_features import create_board_tensor
from experimental.machine_learning.players.reinforcement import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    normalize_action,
)
from experimental.machine_learning.players.minimax import (
    ValueFunctionPlayer,
    VictoryPointPlayer,
)


FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)

DISCOUNT = 0.9

# Every 5 episodes we have ~MINIBATCH_SIZE=1024 samples.
# With batch-size=16k we are likely to hit 1 sample per action(?)
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5_000  # Min number of steps in a memory to start training
MINIBATCH_SIZE = 1024  # How many steps (samples) to use for training
TRAIN_EVERY_N_EPISODES = 1
TRAIN_EVERY_N_STEPS = 100  # catan steps / decisions by agent
UPDATE_MODEL_EVERY_N_TRAININGS = 5  # Terminal states (end of episodes)

# Environment exploration settings
# tutorial settings (seems like 26 hours...)
EPISODES = 20_000
EPSILON_DECAY = 0.99975
# 8 hours process
# EPISODES = 6000
# EPSILON_DECAY = 0.9993
# 2 hours process
# EPISODES = 1500
# EPSILON_DECAY = 0.998
# 30 mins process
# EPISODES = 150
# EPSILON_DECAY = 0.98
# EPISODES = 10_000
epsilon = 1  # not a constant, going to be decayed
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

# TODO: Simple Action Space:
# Hold
# Build Settlement on most production spot vs diff. need number to translate enemy potential to true prod.
# Build City on most production spot.
# Build City on spot that balances production the most.
# Build Road towards more production. (again need to translate potential to true.)
# Buy dev card
# Play Knight to most powerful spot.
# Play Year of Plenty towards most valueable play (city, settlement, dev). Bonus points if use rare resources.
# Play Road Building towards most increase in production.
# Play Monopoly most impactful resource.
# Trade towards most valuable play.

# TODO: Simple State Space:
# Cards in Hand
# Buildable Nodes
# Production
# Num Knights
# Num Roads

DATA_PATH = "data/mcts-playouts"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")


class CatanEnvironment:
    def __init__(self):
        self.game = None
        self.p0 = None

    def playable_actions(self):
        return self.game.state.playable_actions

    def reset(self):
        p0 = Player(Color.BLUE)
        players = [p0, VictoryPointPlayer(Color.RED)]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0_decision()

        return self._get_state()

    def step(self, action_int):
        action = from_action_space(action_int, self.playable_actions())
        self.game.execute(action)

        self._advance_until_p0_decision()
        winning_color = self.game.winning_color()

        new_state = self._get_state()

        # key = player_key(self.game.state, self.p0.color)
        # points = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        # reward = int(winning_color == self.p0.color) * 10 * 1000 + points
        if winning_color is None:
            reward = 0
        elif winning_color == self.p0.color:
            reward = 1
        else:
            reward = -1

        done = winning_color is not None or self.game.state.num_turns > 500
        return new_state, reward, done

    def render(self):
        driver = webdriver.Chrome()
        link = ensure_link(self.game)
        driver.get(link)
        time.sleep(1)
        try:
            driver.close()
        except selenium.common.exceptions.WebDriverException as e:
            print("Exception closing browser. Did you close manually?")

    def _get_state(self):
        sample = create_sample_vector(self.game, self.p0.color, FEATURES)
        # board_tensor = create_board_tensor(self.game, self.p0.color)

        return (sample, None)  # NOTE: each observation/state is a tuple.

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_player().color != self.p0.color
        ):
            self.game.play_tick()  # will play bot


# Agent class
class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # mean = np.load(NORMALIZATION_MEAN_PATH)
        # variance = np.load(NORMALIZATION_VARIANCE_PATH)
        # normalizer_layer = tf.keras.experimental.preprocessing.Normalization(
        #     mean=mean, variance=variance
        # )

        inputs = tf.keras.Input(shape=(NUM_FEATURES,))
        outputs = inputs
        # outputs = normalizer_layer(outputs)
        outputs = BatchNormalization()(outputs)
        # outputs = Dense(352, activation="relu")(outputs)
        # outputs = Dense(256, activation="relu")(outputs)
        outputs = Dense(64, activation="relu")(outputs)
        outputs = Dense(32, activation="relu")(outputs)
        outputs = Dense(units=ACTION_SPACE_SIZE, activation="linear")(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss="mse",
            optimizer=Adam(lr=1e-5),
            metrics=["accuracy"],
        )
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            print("Not enough training data", len(self.replay_memory), MINIBATCH_SIZE)
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = tf.convert_to_tensor([t[0][0] for t in minibatch])
        current_qs_list = self.model.call(current_states).numpy()
        # TODO: Assert randomly distributed.

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = tf.convert_to_tensor([t[3][0] for t in minibatch])
        future_qs_list = self.target_model.call(new_current_states).numpy()

        # Create X, y for training
        X = []
        y = []
        action_ints = list(map(lambda b: b[1], minibatch))
        action_ints_counter = Counter(action_ints)
        sample_weight = []
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state[0])
            y.append(current_qs)
            # w = MINIBATCH_SIZE / (
            #     len(action_ints_counter) * action_ints_counter[action]
            # )
            w = 1 / (action_ints_counter[action])
            sample_weight.append(w)

        # print("Training at", len(self.replay_memory), MINIBATCH_SIZE)
        # print(action_ints_counter)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(y),
            sample_weight=np.array(sample_weight),
            batch_size=MINIBATCH_SIZE,
            epochs=1,
            verbose=0,
            shuffle=False,  # no need since minibatch already was a random sampling
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_MODEL_EVERY_N_TRAININGS:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            print("Updated model!")

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        (sample, board_tensor) = state
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))
        return self.model.call(sample)[0]


# encode action to action_space
def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


# decode action_space to action
def from_action_space(action_int, playable_actions):
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


def epsilon_greedy_policy(playable_actions, qs, epsilon):
    if np.random.random() > epsilon:
        # Create array like [0,0,1,0,0,0,1,...] representing actions in space that are playable
        action_ints = list(map(to_action_space, playable_actions))
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
        mask[action_ints] = 1

        clipped_probas = np.multiply(mask, qs)
        clipped_probas[clipped_probas == 0] = -np.inf

        best_action_int = np.argmax(clipped_probas)
    else:
        # Get random action
        index = random.randrange(0, len(playable_actions))
        best_action = playable_actions[index]
        best_action_int = to_action_space(best_action)

    return best_action_int


# TODO: Fix
# def boltzmann_policy(playable_actions, qs):
#     # Create array like [0,0,1,0,0,0,1,...] representing actions in space that are playable
#     action_ints = list(map(to_action_space, playable_actions))
#     mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
#     mask[action_ints] = 1

#     # Get action from Q table
#     clipped_probas = np.multiply(mask, qs)
#     translation = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)
#     translation[clipped_probas != 0] = abs(clipped_probas.min())
#     translated = clipped_probas + translation
#     if translated.sum() == 0:
#         # HMM... How many times, does this happen?
#         index = random.randrange(0, len(playable_actions))
#         random_action = playable_actions[index]
#         normalized = normalize_action(random_action)
#         action = ACTIONS_ARRAY.index((normalized.action_type, normalized.value))
#     else:
#         normalized_clipped_probas = translated / translated.sum()
#         action = np.random.choice(
#             range(len(ACTIONS_ARRAY)), p=normalized_clipped_probas
#         )

#     return action


DNQ_MODEL = None


class DQNPlayer(Player):
    def __init__(self, color, model_path):
        super(DQNPlayer, self).__init__(color)
        self.model_path = model_path
        global DNQ_MODEL
        # DNQ_MODEL = tf.keras.models.load_model(model_path)
        DNQ_MODEL = DQNAgent().create_model()

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        sample = create_sample_vector(game, self.color, FEATURES)
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))
        qs = DNQ_MODEL.call(sample)[0]

        best_action_int = epsilon_greedy_policy(playable_actions, qs, 0.05)
        best_action = from_action_space(best_action_int, playable_actions)
        return best_action


@click.command()
@click.argument("experiment_name")
def main(experiment_name):
    global epsilon

    env = CatanEnvironment()

    # For stats
    ep_rewards = []

    # For more repetitive results
    random.seed(2)
    np.random.seed(2)
    tf.random.set_seed(2)

    # Ensure models folder
    model_name = f"{experiment_name}-{int(time.time())}"
    models_folder = "experimental/models/"
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)

    agent = DQNAgent()
    metrics_path = f"data/logs/catan-dql/{model_name}"
    output_model_path = models_folder + model_name
    writer = tf.summary.create_file_writer(metrics_path)
    print("Will be writing metrics to", metrics_path)
    print("Will be saving model to", output_model_path)

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            best_action_int = epsilon_greedy_policy(
                env.playable_actions(), agent.get_qs(current_state), epsilon
            )
            new_state, reward, done = env.step(best_action_int)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory(
                (current_state, best_action_int, reward, new_state, done)
            )
            if step % TRAIN_EVERY_N_STEPS == 0:
                agent.train(done)

            current_state = new_state
            step += 1
        if step % TRAIN_EVERY_N_EPISODES == 0:
            agent.train(done)

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if episode % AGGREGATE_STATS_EVERY == 0:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-AGGREGATE_STATS_EVERY:]
            )
            with writer.as_default():
                tf.summary.scalar("avg-reward", average_reward, step=episode)
                tf.summary.scalar("epsilon", epsilon, step=episode)
                writer.flush()

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    print("Saving model to", output_model_path)
    agent.model.save(output_model_path)


if __name__ == "__main__":
    main()
