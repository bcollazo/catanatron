# import gym

# env = gym.make("CartPole-v0")
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())  # take a random action
# env.close()

import gym
import numpy as np

# from rl.callbacks import WandbLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v0")
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
# env.close()
episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))

states = env.observation_space.shape
actions = env.action_space.n

import tensorflow as tf


# def build_model(states, actions):
# inputs = tf.keras.Input(shape=states)
# outputs = inputs
# # outputs = tf.keras.layers.Flatten()(outputs)
# outputs = tf.keras.layers.Dense(units=24, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(units=24, activation="relu")(outputs)
# outputs = tf.keras.layers.Dense(units=1, activation="linear")(outputs)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation="relu", input_shape=states))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


model = build_model(states, actions)
model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e2,
    )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])
dqn.fit(
    env,
    nb_steps=50000,
    visualize=False,
    verbose=1,
    # callbacks=[WandbLogger()],
)


scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history["episode_reward"]))
_ = dqn.test(env, nb_episodes=15, visualize=True)
