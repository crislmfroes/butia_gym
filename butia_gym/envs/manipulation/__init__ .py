import gym

gym.envs.register(
     id='DoRISPickAndPlace-v1',
     entry_point='butia_gym.envs.manipulation.pick_and_place_env:DoRISPickAndPlaceEnv',
     max_episode_steps=50,
)

#register_env('DoRISPickAndPlace-v1', ray_env_creator('DoRISPickAndPlace-v1'))


gym.envs.register(
     id='DoRISPickAndPlaceShaped-v1',
     entry_point='butia_gym.envs.manipulation.pick_and_place_env:DoRISPickAndPlaceEnv',
     max_episode_steps=50,
     kwargs=dict(
          reward_type="shaped",
     )
)

gym.envs.register(
     id='DoRISPickAndPlaceDense-v1',
     entry_point='butia_gym.envs.manipulation.pick_and_place_env:DoRISPickAndPlaceEnv',
     max_episode_steps=50,
     kwargs=dict(
          reward_type="dense",
     )
)
