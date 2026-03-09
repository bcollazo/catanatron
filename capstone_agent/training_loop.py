import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from CapstoneAgent import CapstoneAgent
from action_map import validate as validate_action_mapping
import torch
import gymnasium

OBS_SIZE = 1258
HIDDEN_SIZE = 512
agent = CapstoneAgent(obs_size=OBS_SIZE, hidden_size=HIDDEN_SIZE)
env = gymnasium.make("catanatron/CapstoneCatanatron-v0")
validate_action_mapping()

ROLLOUT_LENGTH = 2048
NUM_UPDATES = 500

obs, info = env.reset()
mask = info["action_mask"]
step = 0

for update in range(NUM_UPDATES):
    for _ in range(ROLLOUT_LENGTH):
        action, log_prob, value = agent.select_action(obs, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["action_mask"]

        agent.store(obs, mask, action, log_prob, reward, value, done)
        step += 1

        if done:
            obs, info = env.reset()
            mask = info["action_mask"]
        else:
            obs, mask = next_obs, next_mask

    with torch.no_grad():
        _, last_value = agent.model(
            torch.FloatTensor(obs).unsqueeze(0),
            torch.FloatTensor(mask).unsqueeze(0),
        )
    agent.train(last_value.item())

    if (update + 1) % 10 == 0:
        print(f"Update {update + 1}/{NUM_UPDATES} complete  (step {step})")

agent.save("capstone_model.pt")
print("Training complete — model saved to capstone_model.pt")
