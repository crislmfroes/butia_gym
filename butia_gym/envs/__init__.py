from butia_gym.envs.manipulation import DoRISPickAndPlaceEnv
from gym.envs.registration import (
    registry,
    register,
)

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    register(
        id=f"DoRISPickAndPlace{suffix}-v1",
        entry_point="butia_gym.envs.manipulation:DoRISPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
