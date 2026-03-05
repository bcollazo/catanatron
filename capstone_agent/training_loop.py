from CapstoneAgent import CapstoneAgent
import torch
import gymnasium

OBS_SIZE = 1258
HIDDEN_SIZE = 512
agent = CapstoneAgent(obs_size=OBS_SIZE, hidden_size=512)
env = gymnasium.make("catanatron/CapstoneCatanatron-v0")

ROLLOUT_LENGTH = 2048  # collect this many steps before updating

state, mask = env.reset()
step = 0

while True:
    action, log_prob, value = agent.select_action(state, mask)
    next_state, next_mask, reward, done = env.step(action)

    agent.store(state, mask, action, log_prob, reward, value, done)
    step += 1

    if done:
        state, mask = env.reset()
    else:
        state, mask = next_state, next_mask

    # Update every N steps
    if step % ROLLOUT_LENGTH == 0:
        _, last_value = agent.model(
            torch.FloatTensor(state).unsqueeze(0),
            torch.FloatTensor(mask).unsqueeze(0)
        )
        agent.learn(last_value.item())