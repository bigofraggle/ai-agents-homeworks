from gymnasium.envs.registration import register

from env import DiscretizedMountainCarEnv

register(
    id="DiscretizedMountainCar-v0",
    entry_point=DiscretizedMountainCarEnv,
)
