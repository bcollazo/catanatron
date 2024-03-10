import numpy as np
import tensorflow as tf
from gymnasium import spaces
import ray
from ray import train, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_tf
from catanatron_gym.envs.catanatron_env import CatanatronEnv

tf1, tf, tfv = try_import_tf()

ray.init(num_cpus=0, local_mode=True)


class ActionMaskEnv(CatanatronEnv):
    """A randomly acting environment that publishes an action-mask each step."""

    def __init__(self, config):
        super().__init__(config)
        # Add action_mask to observations.
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations": self.observation_space,
            }
        )

    def _get_observation(self):
        base_obs = super()._get_observation()
        valid_actions = super().get_valid_actions()

        np_mask = np.zeros(self.action_space.n, dtype=np.int8)
        np_mask[valid_actions] = 1
        return {
            "action_mask": np_mask,
            "observations": base_obs,
        }


# Taken from https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # To visualize, network, put a breakpoint and run the following:
        # from tensorflow.keras.utils import plot_model
        # model = self.internal_model.base_model
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

model_config = {
    "custom_model": "action_mask_model",
    # New format?
    "fcnet_hiddens": [256, 256, 256, 256, 256],
    "fcnet_activation": "relu",
    "vf_share_layers": False,
}

# Had to use TF, because Ray+PyTorch didn't seem to detect my M1 GPU:
# https://discuss.ray.io/t/rllib-pytorch-and-mac-m1-gpus-no-available-node-types-can-fulfill-resource-request/6769/4
# TF was crashing on M1 if using tensorflow-macos==2.15.0 and tensorflow-metal==1.1.0.
# Had to fallback to tensorflow-macos==2.9 and tensorflow-metal==0.5
config = (
    PPOConfig()
    .environment(env=ActionMaskEnv, env_config={"foo": "bar"})
    .framework(framework="tf", eager_tracing=True)
    .training(
        gamma=tune.grid_search([0.99, 0.95, 0.9]),
        lr=tune.grid_search([0.0001]),
        model=model_config,
        shuffle_sequences=True,
    )
    .rollouts(num_envs_per_worker=4)  # TODO?
    # Seems like defaults use the right thing! Use GPU and 6/8 CPUs which sounds good.
)

tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(stop={"episode_reward_mean": 0.9}),
    param_space=config,
)

tuner.fit()

"""
TODO:
- Simple environment (Check Vectorizable).
- Action Masking. DONE.
- Simple Network Architecture. DONE.
- MultiAgent Environment
- Network Architecture(?)

- Was checking PPO config (vf-layers). added shuffle_sequences
"""
