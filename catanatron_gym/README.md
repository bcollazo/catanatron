# catanatron_gym

For reinforcement learning purposes, we provide an Open AI Gym to play 1v1 Catan against a random bot environment. To use:

```
pip install catanatron
```

Make your training loop, ensuring to respect `env.get_valid_actions()`.

```python
env = gym.make("catanatron_gym:catanatron-v0")
observation = env.reset()
for _ in range(1000):
  action = random.choice(env.get_valid_actions()) # your agent here (this takes random actions)

  observation, reward, done, info = env.step(action)
  if done:
      observation = env.reset()
env.close()
```

You can access `env.game.state` and build your own "observation" (features) vector as well.
