import gym
from ray.tune.registry import register_env

def ray_env_creator(env_name):
     def env_creator(env_config):
          return gym.make(env_name, *env_config)
     return env_creator

gym.envs.register(
     id='DoRISPickAndPlace-v1',
     entry_point='butia_gym.envs.manipulation.pick_and_place_env:DoRISPickAndPlaceEnv',
     max_episode_steps=50,
)

gym.envs.register(
    id='DoRISDiverseObjectGrasping-v0',
    entry_point='butia_gym.envs.manipulation.visual_grasp_env:DoRISDiverseObjectEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register_env('DoRISPickAndPlace-v1', ray_env_creator('DoRISPickAndPlace-v1'))
