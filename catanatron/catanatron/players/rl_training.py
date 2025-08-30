import random
import gymnasium
import catanatron.gym
from catanatron.models.player import Player, Color, RandomPlayer


players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
env = gymnasium.make("catanatron/Catanatron-v0", players=players)
observation, info = env.reset()
current_agent_idx = 0

for _ in range(1000):
    action = players[current_agent_idx].decide(env.game, info["valid_actions"])

    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Agent {current_agent_idx} took action {action}, reward: {reward}")
    done = terminated or truncated
    if done:
        observation, info = env.reset()
        current_agent_idx = 0
    else:
        current_agent_idx = (current_agent_idx + 1) % len(players)


print("Game over")
env.close()
