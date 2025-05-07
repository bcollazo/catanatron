from gymnasium.envs.registration import register

register(
    id="catanatron_env/Catanatron-v0",
    entry_point="catanatron_env.envs:CatanatronEnv",
)
