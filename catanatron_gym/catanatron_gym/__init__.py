from gymnasium.envs.registration import register

register(
    id="catanatron_gym/Catanatron-v0",
    entry_point="catanatron_gym.envs:CatanatronEnv",
)
