import os

from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger.unified import UnifiedLogger

from catanatron import RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer

from ray_model import register_model
from ray_env import register_catanatron_env, base_path, experiment_name

register_model()
register_catanatron_env()


# ===== EXPERIMENT
# ray.init()
# config = {
#     "env_config": {
#         "enemies": [
#             RandomPlayer(Color.WHITE),
#         ]
#     },  # config to pass to env class (same as config in CatanatronEnv)
#     "num_workers": 4,
#     "model": {"custom_model": "action_mask_model"},
#     "lr": 1e-4,
# }


def custom_log_creator(custom_path, custom_dir):
    # timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    # logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        # logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        logdir = os.path.join(custom_path, custom_dir)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=1)
    .environment(
        env="ray_catanatron_env",
        env_config={
            "enemies": [
                # RandomPlayer(Color.RED),
                WeightedRandomPlayer(Color.RED),
            ]
        },
    )
    .training(
        lr=1e-4,
        model={
            "custom_model": "action_mask_model",
            "custom_model_config": {
                "fcnet_hiddens": [64, 64, 64, 64, 64, 64],
                "fcnet_activation": "relu",
            },
        },
    )
    .build(logger_creator=custom_log_creator(base_path, experiment_name))
)

# STEPS for re-running:
# 1. Change checkpoint_path here
# 2. Change experiment_name

checkpoint_path = "/Users/bcollazo/ray_results/PPO_ray_catanatron_env_2023-01-22_19-38-14wtmduchc/checkpoint_000096"
checkpoint_path = "/Users/bcollazo/BryanCode/catanatron/logs/PPO_selflearn_64xx6relu/checkpoint_001000"
checkpoint_path = "/Users/bcollazo/BryanCode/catanatron/logs/PPO_selflearn_64xx6relu-2/checkpoint_002000"
# checkpoint_path = ""
if checkpoint_path:
    print("LOADING PREVIOUSLY SAVED CHECKPOINT", checkpoint_path)
    # algo = Algorithm.from_checkpoint(checkpoint_path)
    algo.restore(checkpoint_path)
else:
    print("STARTING TRAINING FROM SCRATCH")

print("STARTING TRAINING")
for i in range(10000):
    print("============", i)
    result = algo.train()
    print(pretty_print(result))

    if (i + 1) % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

print("DONE")
