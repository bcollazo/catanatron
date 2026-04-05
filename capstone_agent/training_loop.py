import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from MainPlayAgent import MainPlayAgent
from Placement.PlacementAgent import PlacementAgent
from CapstoneAgent import CapstoneAgent
from action_map import validate as validate_action_mapping
from device import get_device
import torch
import gymnasium

from CONSTANTS import FEATURE_SPACE_SIZE, MAIN_PLAY_AGENT_HIDDEN_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE
from CONFIG import ROLLOUT_LENGTH, NUM_UPDATES, STORE_FREQUENCY

main_agent = MainPlayAgent(obs_size=FEATURE_SPACE_SIZE, hidden_size=MAIN_PLAY_AGENT_HIDDEN_SIZE)
placement_agent = PlacementAgent(obs_size=FEATURE_SPACE_SIZE, hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE)
env = gymnasium.make("catanatron/CapstoneCatanatron-v0")
agent = CapstoneAgent(placement_agent, main_agent)
validate_action_mapping()

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

    device = get_device()
    with torch.no_grad():
        _, last_value = agent.model(
            torch.FloatTensor(obs).unsqueeze(0).to(device),
            torch.FloatTensor(mask).unsqueeze(0).to(device),
        )
    agent.train(last_value.item())

    if (update + 1) % STORE_FREQUENCY == 0:
        print(f"Update {update + 1}/{NUM_UPDATES} complete  (step {step})")

agent.save("capstone_agent/models/capstone_model.pt", "capstone_agent/models/placement_model.pt")
print("Training complete — models saved to capstone_agent/models/")
