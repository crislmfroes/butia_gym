#!/usr/bin/env bash

rllib train --run=PPO --env='butia_gym.envs.manipulation.visual_grasp_env.DoRISDiverseObjectEnv' --config='{"num_workers": 8, "num_gpus": 1, "output": "../grasping-out", "env_config": {"isDiscrete": False, "width": 84, "height": 84}}' --stop='{"timesteps_total": 1000000}'