from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib import agents
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

import tensorflow as tf

from catanatron_gym.envs.catanatron_env import CatanatronEnv


# https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
class KP0ActionMaskModel(TFModelV2):
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
        **kwargs
    ):
        super(KP0ActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        self.action_embed_model = FullyConnectedNetwork(
            CatanatronEnv.observation_space,
            action_space,
            action_embed_size,
            model_config,
            name + "_action_embedding",
        )
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["state"]}
        )
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


ModelCatalog.register_custom_model("kp_mask", KP0ActionMaskModel)

# ray.init(ignore_reinit_error=True)

# trainer_config = {"model": {"custom_model": "kp_mask"}, "env_config": env_config}
# trainer = agents.ppo.PPOTrainer(env="Knapsack-v0", config=trainer_config)


# Using: https://github.com/ray-project/ray/issues/7983
def train_ppo(config, reporter):
    agent = PPOTrainer(config)
    # agent.restore("/path/checkpoint_41/checkpoint-41")  # continue training
    i = 0
    while True:
        result = agent.train()
        if reporter is None:
            continue
        else:
            reporter(**result)
        if i % 10 == 0:  # save every 10th training iteration
            checkpoint_path = agent.save()
            print(checkpoint_path)
        i += 1
        # you can also change the curriculum here


config = {
    "env": CatanatronEnv,
    "num_workers": 4,
    # "model": {"custom_model": "kp_mask"},
}
print(config)
# trainingSteps = 1000000
# trainingSteps = 1000
# trials = tune.run(
#     train_ppo,
#     config=config,
#     stop={
#         "training_iteration": trainingSteps,
#     },
# )
# breakpoint()
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
trainer = ppo.PPOTrainer(config=config)

# Can optionally call trainer.restore(path) to load a checkpoint.
for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(i, pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
# tune.run(
#     "PPO",
#     config=config,
# )
