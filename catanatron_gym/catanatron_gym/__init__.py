from gymnasium.envs.registration import register

register(
    id="catanatron-v0",
    entry_point="catanatron_gym.envs:CatanatronEnv",
)
