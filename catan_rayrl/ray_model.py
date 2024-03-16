from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
import numpy as np

from catanatron_gym.envs.catanatron_env import CatanatronEnv


tf = try_import_tf()
tf, tf2, tf_version = tf

# ===== MODEL
class ActionMaskModel(TFModelV2):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(11,),
        action_embed_size=5,
        *args,
        **kwargs,
    ):
        super(ActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        print("Constructing ActionMaskModel", model_config)
        self.action_space = action_space
        # self.action_embed_model = FullyConnectedNetwork(
        #     spaces.Box(0, 1, shape=true_obs_shape),
        #         action_space, action_embed_size,
        #     model_config, name + "_action_embedding")
        obs_space = CatanatronEnv.observation_space
        self.model = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_action_embedding",
        )

    def forward(self, input_dict, state, seq_lens):
        mask = input_dict["obs"]["action_mask"]
        avail_actions = np.ones(self.action_space.n, dtype=np.int16)

        # avail_actions = input_dict["obs"]["avail_actions"]
        # action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.model({"obs": input_dict["obs"]["actual_obs"]})
        # action_embedding, _ = self.model({
        #     "obs": input_dict["obs"]["actual_obs"]})

        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf.maximum(tf.log(mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.model.value_function()

def register_model():
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)