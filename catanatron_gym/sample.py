import random
import gymnasium as gym

env = gym.make("catanatron_gym:catanatron-v0")
observation, info = env.reset()
for _ in range(1000):
    action = random.choice(
        env.get_valid_actions()
    )  # your agent here (this takes random actions)

    observation, reward, done, info = env.step(action)
    if done:
        observation, info = env.reset()
env.close()
