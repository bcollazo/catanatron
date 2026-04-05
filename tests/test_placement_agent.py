import numpy as np
import torch

from capstone_agent.PlacementAgent import PlacementAgent
from capstone_agent.placement_action_space import (
    PlacementPrompt,
    capstone_action_to_local,
    capstone_mask_to_local_mask,
    infer_placement_prompt,
    local_action_to_capstone,
)
from capstone_agent.placement_features import (
    COMPACT_PLACEMENT_FEATURE_SIZE,
    get_compact_placement_observation,
    project_capstone_to_compact_placement,
)
from catanatron.gym.envs.capstone_env import CapstoneCatanatronEnv


def test_compact_placement_features_match_projected_capstone_obs():
    env = CapstoneCatanatronEnv()
    observation, info = env.reset(seed=42)

    checked_states = 0
    while info["is_initial_build_phase"]:
        direct = get_compact_placement_observation(
            env.game,
            self_color=env.self_player.color,
            opp_color=env.opp_color,
        )
        projected = project_capstone_to_compact_placement(
            observation, info["action_mask"]
        )

        assert direct.shape == (COMPACT_PLACEMENT_FEATURE_SIZE,)
        np.testing.assert_allclose(direct, projected, atol=1e-6)

        observation, _, terminated, truncated, info = env.step(info["valid_actions"][0])
        checked_states += 1
        assert not terminated
        assert not truncated

    assert checked_states == 4
    env.close()


def test_placement_action_space_maps_local_and_global_indices():
    env = CapstoneCatanatronEnv()
    _, info = env.reset(seed=42)

    prompt = infer_placement_prompt(info["action_mask"])
    assert prompt == PlacementPrompt.SETTLEMENT
    local_mask = capstone_mask_to_local_mask(info["action_mask"], prompt)
    local_valid = np.where(local_mask > 0.5)[0]
    mapped_valid = {
        local_action_to_capstone(prompt, idx)
        for idx in local_valid
    }
    assert mapped_valid == set(info["valid_actions"])

    sample_action = info["valid_actions"][0]
    local_idx = capstone_action_to_local(prompt, sample_action)
    assert local_action_to_capstone(prompt, local_idx) == sample_action

    _, _, _, _, info = env.step(sample_action)
    prompt = infer_placement_prompt(info["action_mask"])
    assert prompt == PlacementPrompt.ROAD
    local_mask = capstone_mask_to_local_mask(info["action_mask"], prompt)
    local_valid = np.where(local_mask > 0.5)[0]
    mapped_valid = {
        local_action_to_capstone(prompt, idx)
        for idx in local_valid
    }
    assert mapped_valid == set(info["valid_actions"])

    env.close()


def test_compact_placement_agent_returns_legal_capstone_actions():
    torch.manual_seed(0)
    env = CapstoneCatanatronEnv()
    agent = PlacementAgent()

    observation, info = env.reset(seed=7)
    placement_steps = 0

    while info["is_initial_build_phase"]:
        mask = info["action_mask"]
        action, log_prob, value = agent.select_action(observation, mask)
        assert action in info["valid_actions"]
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

        next_observation, reward, terminated, truncated, next_info = env.step(action)
        agent.store(
            observation,
            mask,
            action,
            log_prob,
            reward,
            value,
            terminated or truncated,
        )

        observation = next_observation
        info = next_info
        placement_steps += 1

    assert placement_steps == 4
    agent.train(last_value=0.0)
    assert len(agent.buffer.states) == 0
    env.close()
