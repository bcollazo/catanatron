from gymnasium.envs.registration import register

register(
    id="catanatron_env/GridWorld-v0",
    entry_point="catanatron_env.envs:GridWorldEnv",
)
