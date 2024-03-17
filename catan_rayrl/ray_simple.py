import random
from typing import List, Optional, Union

from ray.tune.search.hyperopt import HyperOptSearch
import numpy as np
import tensorflow as tf
import tree  # pip install dm_tree
from gymnasium import spaces
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF2Policy
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec

# The new RLModule / Learner API
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

# from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType
from ray.tune.schedulers import ASHAScheduler

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.map import build_map
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)
from catanatron_gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE,
    HIGH,
    NUM_FEATURES,
    CatanatronEnv,
    from_action_space,
    simple_reward,
    to_action_space,
)
from catanatron_gym.features import create_sample, get_feature_ordering

tf1, tf, tfv = try_import_tf()


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
        orig_space.get("BLUE", orig_space.get("RED", orig_space))
        simple_orig_space = orig_space.get("BLUE", orig_space.get("RED", orig_space))
        if not (
            isinstance(simple_orig_space, spaces.Dict)
            and "action_mask" in simple_orig_space.spaces
            and "observations" in simple_orig_space.spaces
        ):
            print(obs_space)
            assert False

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            simple_orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        if "obs" not in input_dict or "action_mask" not in input_dict["obs"]:
            print(input_dict)
            assert False
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


class CatanatronMultiAgentEnv(MultiAgentEnv):
    metadata = {"render_modes": []}

    _agent_ids = ["BLUE", "RED"]
    simple_action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    action_space = spaces.Dict(
        {"BLUE": simple_action_space, "RED": simple_action_space}
    )
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    simple_observation_space = spaces.Box(
        low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=np.float32
    )
    masked_simple_observation_space = spaces.Dict(
        {
            "action_mask": spaces.Box(
                0.0, 1.0, shape=(simple_action_space.n,), dtype=np.float32
            ),
            "observations": simple_observation_space,
        }
    )
    observation_space = spaces.Dict(
        {
            "BLUE": masked_simple_observation_space,
            "RED": masked_simple_observation_space,
        }
    )
    reward_range = (-1, 1)

    def __init__(self, config=None):
        super().__init__()
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        # self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=np.float32
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float32
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.simple_observation_space = mixed
        else:
            self.simple_observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float32
            )

        self.masked_simple_observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(
                    0.0, 1.0, shape=(self.simple_action_space.n,), dtype=np.float32
                ),
                "observations": self.simple_observation_space,
            }
        )
        self.observation_space = spaces.Dict(
            {
                "BLUE": self.masked_simple_observation_space,
                "RED": self.masked_simple_observation_space,
            }
        )
        self.reset()

    def step(self, action_dict):
        assert len(action_dict) == 1
        (color, action) = action_dict.popitem()
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            # TODO: Handle
            self.invalid_actions_count += 1

            observation = self._get_observation(Color[color])
            winning_color = self.game.winning_color()
            terminated = winning_color is not None
            # should be impossible to win or terminate with an invalid action
            assert winning_color is None
            assert not terminated
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )

            infos = {c.value: {} for c in self.game.state.colors}
            infos = {}
            infos[color] = dict(valid_actions=self.get_valid_actions())
            return (
                {color: observation},
                {color: self.invalid_action_reward},
                {color: terminated, "__all__": terminated},
                {color: truncated, "__all__": truncated},
                infos,
            )

        assert catan_action.color.value == color
        self.game.execute(catan_action)

        current_color = self.game.state.current_color()
        observation = self._get_observation(current_color)
        infos = {c.value: {} for c in self.game.state.colors}
        infos = {}
        infos[current_color.value] = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, current_color)
        # TODO: Investigate
        # if terminated and reward <= 0:
        #     breakpoint()
        # if terminated and winning_color != current_color:
        #     breakpoint()

        return (
            {current_color.value: observation},
            {current_color.value: reward},
            {current_color.value: terminated, "__all__": terminated},
            {current_color.value: truncated, "__all__": truncated},
            infos,
        )

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self.invalid_actions_count = 0

        current_color = self.game.state.current_color()
        observation = self._get_observation(current_color)
        infos = {c.value: {} for c in self.game.state.colors}
        infos = {}
        infos[current_color.value] = dict(valid_actions=self.get_valid_actions())

        return {current_color.value: observation}, infos

    def _get_observation(self, color):
        sample = create_sample(self.game, color)
        valid_actions = self.get_valid_actions()
        np_mask = np.zeros(self.simple_action_space.n, dtype=np.float32)
        np_mask[valid_actions] = 1
        if self.representation == "mixed":
            board_tensor = create_board_tensor(self.game, color, channels_first=True)
            numeric = np.array([np.float32(sample[i]) for i in self.numeric_features])

            return {
                "action_mask": np_mask,
                "observations": {"board": board_tensor, "numeric": numeric},
            }
        return {
            "action_mask": np_mask,
            "observations": np.array([np.float32(sample[i]) for i in self.features]),
        }

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))


class ValidRandomPolicy(Policy):
    """Hand-coded policy that returns random (but valid) actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space_for_sampling = self.action_space

    @override(Policy)
    def init_view_requirements(self):
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        **kwargs,
    ):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # obs_batch_size = len(tree.flatten(obs_batch)[0])
        actions = [
            np.random.choice(np.where(mask == 1)[0])
            for mask in obs_batch["action_mask"]
        ]
        return (actions, [], {})

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        **kwargs,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        self.consumed_rewards = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        # No need to check on every hist_stats.
        # Take last 25 training games as evaluation
        main_rew = result["hist_stats"].pop("policy_main_reward")
        if self.current_opponent == 0:
            latest_enemy_rew = result["hist_stats"].pop("policy_random_reward")
        else:
            latest_enemy_rew = result["hist_stats"].pop(
                f"policy_main_v{self.current_opponent}_reward"
            )
        if len(latest_enemy_rew) < MIN_TO_CHECK_SELF_TRAINING:
            return
        to_check = main_rew[-MIN_TO_CHECK_SELF_TRAINING:]

        win_rate = sum(to_check) / len(to_check)
        result["win_rate"] = win_rate
        print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > WIN_RATE_THRESHOLD:
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                return (
                    "main"
                    if agent_id == "BLUE"
                    else "main_v{}".format(
                        np.random.choice(list(range(1, self.current_opponent + 1)))
                    )
                )

            main_policy = algorithm.get_policy("main")
            if algorithm.config._enable_new_api_stack:
                new_policy = algorithm.add_policy(
                    policy_id=new_pol_id,
                    policy_cls=type(main_policy),
                    policy_mapping_fn=policy_mapping_fn,
                    module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model),
                )
            else:
                new_policy = algorithm.add_policy(
                    policy_id=new_pol_id,
                    policy_cls=type(main_policy),
                    policy_mapping_fn=policy_mapping_fn,
                    observation_space=main_policy.observation_space,
                    action_space=main_policy.action_space,
                    # config=config,
                    # policy_state=policy_state,
                    # policies_to_train=policies_to_train,
                    # module_spec=module_spec,
                )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            main_state = main_policy.get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            algorithm.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 2


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """We can always assign BLUE to main, since the environment will
    shuffle seat assignment in .reset() anyways."""
    return "main" if agent_id == "BLUE" else "random"


num_layers = [5, 10, 20]
num_nodes = [64, 128, 256, 512]
architectures = [[N] * L for N in num_nodes for L in num_layers]

# num_layers = 10
# num_nodes = 256
# architectures = [[num_nodes] * num_layers]

WIN_RATE_THRESHOLD = 0.7
MIN_TO_CHECK_SELF_TRAINING = 25

# Had to use TF, because Ray+PyTorch didn't seem to detect my M1 GPU:
# https://discuss.ray.io/t/rllib-pytorch-and-mac-m1-gpus-no-available-node-types-can-fulfill-resource-request/6769/4
# TF was crashing on M1 if using tensorflow-macos==2.15.0 and tensorflow-metal==1.1.0.
# Had to fallback to tensorflow-macos==2.9 and tensorflow-metal==0.5
config = (
    PPOConfig()
    # .environment(env=ActionMaskEnv, env_config={"foo": "bar"})
    .framework(framework="tf2", eager_tracing=True)
    .training(
        gamma=tune.choice([0.99, 0.999, 0.9999]),
        lr=tune.choice([0.0001, 0.00001]),
        model={
            "custom_model": "action_mask_model",
            "fcnet_hiddens": tune.choice(architectures),
            "fcnet_activation": "relu",
            "vf_share_layers": True,
        },
        shuffle_sequences=True,
        train_batch_size=20000,  # at least 20 episodes...
        sgd_minibatch_size=1024,
        vf_loss_coeff=tune.choice([0.5, 1]),
    )
    # .rollouts(num_rollout_workers=4, num_envs_per_worker=1)
    .experimental(_disable_preprocessor_api=True)
    .callbacks(SelfPlayCallback)
    .environment(env=CatanatronMultiAgentEnv, env_config={"foo": "bar"})
    .multi_agent(
        policies={
            "main": PolicySpec(),
            "random": PolicySpec(policy_class=ValidRandomPolicy),
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["main"],  # Always just train the "main" policy.
    )
    # Seems like defaults use the right thing! Use GPU and 6/8 CPUs which sounds good.
    # .resources()
)
# ray.init(num_cpus=0, local_mode=True)

mlflow_tracking_uri = "http://127.0.0.1:8080"

hyperopt_search = HyperOptSearch(metric="episode_len_mean", mode="min")

RESUME = "/Users/bcollazo/ray_results/PPO_2024-03-17_12-18-57"
RESUME = ""
trainable = "PPO"
tuner = tune.Tuner(
    trainable,
    run_config=train.RunConfig(
        stop={"episode_len_mean": 300, "time_total_s": 30 * 60},
        name="mlflow",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=f"Catanatron",
                save_artifact=True,
            )
        ],
    ),
    tune_config=tune.TuneConfig(
        num_samples=10,
        scheduler=ASHAScheduler(metric="episode_len_mean", mode="min"),
        search_alg=hyperopt_search,
    ),
    param_space=config,
)
if RESUME != "":
    tuner.restore(path=RESUME, trainable=trainable)

results = tuner.fit()
breakpoint()

"""
TODO:
- Simple environment. DONE.
- Action Masking. DONE.
- Simple Network Architecture. DONE.
- MultiAgent Environment: https://github.com/ray-project/ray/blob/master/rllib/examples/self_play_with_open_spiel.py. DONE.
- Allow rep=mixed as param. Network Architecture(?). 
- Check running from CLI.
- Find stop criterion
- Scale simple experiment in AWS.
- Scale

- Was checking PPO config (vf-layers). added shuffle_sequences
ray-rl/ray_simple.py:66: DeprecationWarning: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.
  self.internal_model = FullyConnectedNetwork(
WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
"""
