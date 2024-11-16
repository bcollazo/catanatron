import torch as th
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN to process the board observations.
    :param observation_space: (gym.Space)
    :param cnn_arch: List of integers specifying the number of filters in each Conv layer.
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_arch,
        features_dim: int = 256,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space["board"].shape[0]

        layers = []
        in_channels = n_input_channels
        for out_channels in cnn_arch:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Compute the number of features after CNN
        with th.no_grad():
            sample_board = th.as_tensor(
                observation_space.sample()["board"][None]
            ).float()
            n_flatten = self.cnn(sample_board).shape[1]

        n_numeric_features = observation_space["numeric"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_numeric_features, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        board_features = self.cnn(observations["board"])
        concatenated_tensor = th.cat([board_features, observations["numeric"]], dim=1)
        return self.linear(concatenated_tensor)
