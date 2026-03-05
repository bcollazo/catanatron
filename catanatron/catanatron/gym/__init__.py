from gymnasium.envs.registration import register

register(
    id="catanatron/Catanatron-v0",
    entry_point="catanatron.gym.envs:CatanatronEnv",
)

register(
    id="catanatron/CapstoneCatanatron-v0",
    entry_point="catanatron.gym.envs.capstone_env:CapstoneCatanatronEnv",
)
