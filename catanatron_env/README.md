# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).


## Installation

To install your new environment, run the following commands:

```{shell}
cd catanatron_env
pip install -e .
```

