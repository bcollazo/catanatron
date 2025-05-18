import gymnasium
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import catanatron.gym


def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])


# Init Environment and Model
env = gymnasium.make("catanatron/Catanatron-v0")
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train
model.learn(total_timesteps=10_000)
