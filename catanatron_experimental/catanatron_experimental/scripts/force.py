import numpy as np
from tensorforce import Agent, Environment, Runner
from tensorforce.environments import OpenAIGym


class CustomEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type="float", shape=(8,))

    def actions(self):
        return dict(type="int", num_values=4)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = np.random.random() < 0.5
        reward = np.random.random()
        return next_state, terminal, reward


environment = OpenAIGym(
    "CartPole",
    visualize=True,
    #  visualize_directory='./visualize',
)

# Pre-defined or custom environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500
# )


class BryanAgent(Agent):
    def act(
        self,
        states,
        internals=None,
        parallel=0,
        independent=False,
        deterministic=True,
        evaluation=None,
    ):
        super().act(states, internals, parallel, independent, deterministic, evaluation)
        print("HERE")


# Instantiate a Tensorforce agent
agent = BryanAgent.create(
    agent="tensorforce",
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit="timesteps", batch_size=64),
    optimizer=dict(type="adam", learning_rate=3e-4),
    policy=dict(network="auto"),
    objective="policy_gradient",
    reward_estimation=dict(horizon=20),
)


# ===== OBSERVE ACT
# Train for 100 episodes
N = 50
for episode in range(N):
    # Episode using act and observe
    states = environment.reset()
    terminal = False
    sum_rewards = 0.0
    num_updates = 0
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        num_updates += agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward
    print("Episode {}: return={} updates={}".format(episode, sum_rewards, num_updates))

# Evaluate for 100 episodes
N = 10
sum_rewards = 0.0
for _ in range(N):
    print("Evaluating", _)
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward
print("Mean evaluation return:", sum_rewards / N)

# =====Runner
# runner = Runner(
#     agent=agent,
#     environment=environment
# )
# runner.run(num_episodes=100)
# runner.run(num_episodes=100, evaluation=True)
# runner.close()

# ==== Simple Train
# for _ in range(300):
#     # Initialize episode
#     states = environment.reset()
#     terminal = False
#     print("Episode", _)
#     while not terminal:
#         # Episode timestep
#         actions = agent.act(states=states)
#         states, terminal, reward = environment.execute(actions=actions)
#         agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()
print("HERE")
