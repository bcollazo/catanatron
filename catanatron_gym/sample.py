import random
import gymnasium as gym


env = gym.make("catanatron_gym:catanatron-v1")
observation, info = env.reset()
for _ in range(1000):
    # your agent here (this takes random actions)
    action = random.choice(env.unwrapped.get_valid_actions())
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        observation, info = env.reset()
env.close()
