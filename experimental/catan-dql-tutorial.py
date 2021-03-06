import sys, traceback
from pathlib import Path

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os

from catanatron.game import Game
from catanatron.models.player import Color, Player
from experimental.machine_learning.features import (
    create_sample_vector,
    get_feature_ordering,
)
from experimental.machine_learning.board_tensor_features import create_board_tensor
from experimental.machine_learning.players.reinforcement import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    normalize_action,
)


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

NUM_FEATURES = len(FEATURES)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = f"mbs={MINIBATCH_SIZE}"
TRAIN_EVERY = 4  # catan steps / decisions by agent
DATA_PATH = "data/mcts-playouts"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")

# Environment settings
# tutorial settings (seems like 26 hours...)
EPISODES = 20_000
EPSILON_DECAY = 0.99975
# 8 hours process
EPISODES = 6000
EPSILON_DECAY = 0.9993
# 30 mins process
EPISODES = 150
EPSILON_DECAY = 0.98

EPISODES = 10_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False


class CatanEnv:
    def __init__(self):
        self.game = None
        self.p0 = None
        self.currently_playable = None

    def reset(self):
        p0 = Player(Color.WHITE)
        players = [
            p0,
            Player(Color.RED),
            Player(Color.BLUE),
            Player(Color.ORANGE),
        ]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0_decision()

        state = self._get_state()
        return state

    def step(self, action):
        # Take any action in the playable, that matches ACTIONS_ARRAY prototype
        (action_type, value) = ACTIONS_ARRAY[action]
        catan_action = None
        for a in self.currently_playable:
            normalized = normalize_action(a)
            if normalized.action_type == action_type and normalized.value == value:
                catan_action = a
                break
        if catan_action is None:
            print("Couldnt find", action, action_type, value, "in:")
            print(self.currently_playable)
            breakpoint()
        else:
            self.game.execute(catan_action)

        self._advance_until_p0_decision()
        winning_color = self.game.winning_color()

        new_state = self._get_state()
        reward = int(winning_color == self.p0.color)
        done = winning_color is not None  # TODO: Or too many turns.
        return new_state, reward, done

    def render(self):
        pass

    def _get_state(self):
        sample = create_sample_vector(self.game, self.p0.color, FEATURES)
        # board_tensor = create_board_tensor(self.game, self.p0.color)

        return (sample, None)  # NOTE: each observation/state is a tuple.

    def _advance_until_p0_decision(self):
        while self.game.winning_player() is None:
            # This is basically an adapted self.game.play_tick()
            player, action_prompt = self.game.pop_from_queue()
            self.currently_playable = self.game.playable_actions(player, action_prompt)
            if player.color == self.p0.color and len(self.currently_playable) > 1:
                break
            else:
                # Assume enemies play randomly
                index = random.randrange(0, len(self.currently_playable))
                random_action = self.currently_playable[index]
                self.game.execute(random_action)


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
        # outputs = BatchNormalization()(outputs)
        # outputs = Dense(352, activation=tf.nn.relu)(outputs)
        # outputs = Dense(64, activation=tf.nn.relu)(outputs)
        # outputs = Dense(32, activation=tf.nn.relu)(outputs)
        outputs = Dense(32, activation="relu")(outputs)
        outputs = Dense(units=ACTION_SPACE_SIZE, activation="linear")(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=["mae"])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = tf.convert_to_tensor([t[0][0] for t in minibatch])
        current_qs_list = self.model.call(current_states).numpy()

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = tf.convert_to_tensor([t[3][0] for t in minibatch])
        future_qs_list = self.target_model.call(new_current_states).numpy()

        X = []
        y = []

        # Now we need to enumerate our batches
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

            # # account for class imbalance.
            # # 38.6% of actions are roll (action=0), 27% are end turn (action=5557)
            # if action == 0 or action == ACTION_SPACE_SIZE - 1:
            #     new_q /= 35  # should only happen for `not done` code.

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state[0])
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        (sample, board_tensor) = state
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))
        return self.model.call(sample)[0]


def epsilon_greedy_policy(env, agent, current_state):
    # This part stays mostly the same, the change is to query a model for Q values
    if np.random.random() > epsilon:
        # Create array like [0,0,1,0,0,0,1,...] representing possible actions
        normalized_playable = [normalize_action(a) for a in env.currently_playable]
        possibilities = [(a.action_type, a.value) for a in normalized_playable]
        possible_indices = [ACTIONS_ARRAY.index(x) for x in possibilities]
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
        mask[possible_indices] = 1

        # Get action from Q table
        clipped_probas = np.multiply(mask, agent.get_qs(current_state))
        clipped_probas[clipped_probas == 0] = -np.inf
        action = np.argmax(clipped_probas)
    else:
        # Get random action
        index = random.randrange(0, len(env.currently_playable))
        random_action = env.currently_playable[index]
        normalized = normalize_action(random_action)
        action = ACTIONS_ARRAY.index((normalized.action_type, normalized.value))

    return action


def boltzmann_policy(env, agent, current_state):
    # Create array like [0,0,1,0,0,0,1,...] representing possible actions
    normalized_playable = [normalize_action(a) for a in env.currently_playable]
    possibilities = [(a.action_type, a.value) for a in normalized_playable]
    possible_indices = [ACTIONS_ARRAY.index(x) for x in possibilities]
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
    mask[possible_indices] = 1

    # Get action from Q table
    clipped_probas = np.multiply(mask, agent.get_qs(current_state))
    translation = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)
    translation[clipped_probas != 0] = abs(clipped_probas.min())
    translated = clipped_probas + translation
    if translated.sum() == 0:
        # HMM... How many times, does this happen?
        index = random.randrange(0, len(env.currently_playable))
        random_action = env.currently_playable[index]
        normalized = normalize_action(random_action)
        action = ACTIONS_ARRAY.index((normalized.action_type, normalized.value))
    else:
        normalized_clipped_probas = translated / translated.sum()
        action = np.random.choice(
            range(len(ACTIONS_ARRAY)), p=normalized_clipped_probas
        )

    return action


def main():
    global epsilon

    env = CatanEnv()

    # For stats
    ep_rewards = [0]
    ep_vps = [0]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Create models folder
    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    writer = tf.summary.create_file_writer(f"logs/catan-dql/{int(time.time())}")

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        try:
            done = False
            while not done:
                action = epsilon_greedy_policy(env, agent, current_state)
                new_state, reward, done = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    env.render()

                # Every step we update replay memory and train main network
                agent.update_replay_memory(
                    (current_state, action, reward, new_state, done)
                )
                if step % TRAIN_EVERY == 0:  # train only every TRAIN_EVERY steps
                    agent.train(done)

                current_state = new_state
                step += 1
        except Exception:
            print("Exception in user code:")
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
            breakpoint()
        # Note: We record around 100 decisions per game to replay_memory.

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        ep_vps.append(env.game.players_by_color[Color.WHITE].actual_victory_points)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-AGGREGATE_STATS_EVERY:]
            )
            average_vp = sum(ep_vps[-AGGREGATE_STATS_EVERY:]) / len(
                ep_vps[-AGGREGATE_STATS_EVERY:]
            )
            with writer.as_default():
                tf.summary.scalar("avg-reward", average_reward, step=episode)
                tf.summary.scalar("avg-vp", average_vp, step=episode)
                tf.summary.scalar("epsilon", epsilon, step=episode)
                writer.flush()

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    path = f"models/{MODEL_NAME}__{int(time.time())}.model"
    print("Saving model to", path)
    agent.model.save(path)


if __name__ == "__main__":
    main()
