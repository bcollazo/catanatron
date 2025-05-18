from collections import deque

import gymnasium
import numpy as np

import catanatron.gym


# Initialize environment, buffer and episode_start
num_envs = 4
envs = gymnasium.vector.AsyncVectorEnv(
    [lambda: gymnasium.make("catanatron/Catanatron-v0") for _ in range(num_envs)]
)
replay_buffer = deque(maxlen=100)
episode_start = np.zeros(envs.num_envs, dtype=bool)

observations, infos = envs.reset()
for i in range(1000):  # Training loop
    # Policy would go here, for now choose random action for each env
    actions = [
        np.random.choice(infos["valid_actions"][j]) for j in range(envs.num_envs)
    ]

    next_observations, rewards, terminations, truncations, infos = envs.step(actions)

    # Add to replay buffer
    for i in range(envs.num_envs):
        if not episode_start[i]:
            replay_buffer.append(
                (
                    observations[i],
                    actions[i],
                    rewards[i],
                    terminations[i],
                    next_observations[i],
                )
            )

    # update observation and if episode starts
    observations = next_observations
    episode_start = np.logical_or(terminations, truncations)
envs.close()
