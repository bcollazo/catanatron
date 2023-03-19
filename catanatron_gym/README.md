# Catanatron Gym

For reinforcement learning purposes, we provide an Open AI Gym environment. To use:

```
pip install catanatron_gym
```

Make your training loop, ensuring to respect `env.get_valid_actions()`.

```python
import random
import gymnasium as gym

env = gym.make("catanatron_gym:catanatron-v0")
observation, info = env.reset()
for _ in range(1000):
  action = random.choice(env.get_valid_actions()) # your agent here (this takes random actions)

  observation, reward, done, info = env.step(action)
  if done:
      observation, info = env.reset()
env.close()
```

For `action` documentation see [here](https://catanatron.readthedocs.io/en/latest/catanatron_gym.envs.html#catanatron_gym.envs.catanatron_env.CatanatronEnv.action_space).

For `observation` documentation see [here](https://catanatron.readthedocs.io/en/latest/catanatron_gym.envs.html#catanatron_gym.envs.catanatron_env.CatanatronEnv.observation_space).

You can access `env.game.state` and build your own "observation" (features) vector as well.

## Stable-Baselines3 Example

Catanatron works well with SB3, and better with the Maskable models of the [SB3 Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html) repo. Here a small example of how it may work.

```python
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])


# Init Environment and Model
env = gym.make("catanatron_gym:catanatron-v0")
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train
model.learn(total_timesteps=1_000_000)
```

## Configuration

You can also configure what map to use, how many vps to win, among other variables in the environment,
with the `config` keyword argument. See source for details.

```python
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer


def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100

# 3-player catan on a "Mini" map (7 tiles) until 6 points.
env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "map_type": "MINI",
        "vps_to_win": 6,
        "enemies": [WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.ORANGE)],
        "reward_function": my_reward_function,
        "representation": "mixed",
    },
)
```
