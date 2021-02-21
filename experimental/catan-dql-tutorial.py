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
from catanatron.models.player import RandomPlayer, Color, SimplePlayer, Player
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

# from catanatron.models.enums import Co

NUM_FEATURES = len(get_feature_ordering())

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 256  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = "2x256"
# MIN_REWARD = 0.9  # For model save
# MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.0
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False


class CatanEnv:
    OBSERVATION_SPACE_VALUES = (10, 10, 3)

    def __init__(self):
        self.game = None
        self.p0 = None
        self.currently_playable = None

    def reset(self):
        p0 = Player(Color.ORANGE)
        players = [
            p0,
            Player(Color.RED),
            Player(Color.BLUE),
            Player(Color.WHITE),
        ]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0()

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

        self._advance_until_p0()
        winning_color = self.game.winning_color()

        new_state = self._get_state()
        reward = int(winning_color == self.p0.color)
        done = winning_color is not None  # TODO: Or too many turns.
        return new_state, reward, done

    def render(self):
        pass

    def _get_state(self):
        sample = create_sample_vector(self.game, self.p0.color)
        board_tensor = create_board_tensor(self.game, self.p0.color)

        return (sample, board_tensor)  # each observation/state is a tuple.

    def _advance_until_p0(self):
        while self.game.winning_player() is None:
            # This is basically an adapted self.game.play_tick()
            player, action_prompt = self.game.pop_from_queue()
            self.currently_playable = self.game.playable_actions(player, action_prompt)
            if player.color == self.p0.color:
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
        inputs = tf.keras.Input(shape=(NUM_FEATURES,))
        outputs = inputs
        # outputs = BatchNormalization()(outputs)
        # outputs = Dense(352, activation=tf.nn.relu)(outputs)
        # outputs = Dense(320, activation=tf.nn.relu)(outputs)
        # outputs = Dense(160, activation=tf.nn.relu)(outputs)
        # outputs = Dense(512, activation=tf.nn.relu)(outputs)
        # outputs = Dense(352, activation=tf.nn.relu)(outputs)
        # outputs = Dense(64, activation=tf.nn.relu)(outputs)
        # outputs = Dense(32, activation=tf.nn.relu)(outputs)
        outputs = Dense(32, activation="relu")(outputs)
        outputs = Dense(units=ACTION_SPACE_SIZE, activation="linear")(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
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
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = tf.convert_to_tensor([t[3][0] for t in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

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

            # account for class imbalance.
            # 38.6% of actions are roll (action=0), 27% are end turn (action=5557)
            if action == 0 or action == ACTION_SPACE_SIZE - 1:
                new_q /= 35  # should only happen for `not done` code.

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
        return self.model.predict(sample)[0]


def main():
    global epsilon

    env = CatanEnv()

    # For stats
    ep_rewards = [0]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):

        # Update tensorboard step every episode
        # agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        print("Starting EPISODE", episode)
        try:
            done = False
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Create array like [0,0,1,0,0,0,1,...] representing possible actions
                    normalized_playable = [
                        normalize_action(a) for a in env.currently_playable
                    ]
                    possibilities = [
                        (a.action_type, a.value) for a in normalized_playable
                    ]
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
                    action = ACTIONS_ARRAY.index(
                        (normalized.action_type, normalized.value)
                    )

                new_state, reward, done = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    env.render()

                # Every step we update replay memory and train main network
                agent.update_replay_memory(
                    (current_state, action, reward, new_state, done)
                )
                if step % 4 == 0:  # train only every 4 steps
                    agent.train(done)

                current_state = new_state
                step += 1
        except Exception as e:
            print("ERROR", e)
            breakpoint()
        print(
            "DONE WITH EPISODE",
            episode,
            len(agent.replay_memory),
            len(env.game.actions),
        )

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-AGGREGATE_STATS_EVERY:]
            )
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            # agent.tensorboard.update_stats(
            #     reward_avg=average_reward,
            #     reward_min=min_reward,
            #     reward_max=max_reward,
            #     epsilon=epsilon,
            # )
            print("avg_reward", average_reward)
            print("min_reward", min_reward)
            print("max_reward", max_reward)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    print("Saving model")
    agent.model.save(f"models/{MODEL_NAME}__{int(time.time())}.model")


if __name__ == "__main__":
    main()
