from gymnasium.envs.registration import register

register(
    id="catanatron/Catanatron-v0",
    entry_point="catanatron.gym.envs:CatanatronEnv",
)
