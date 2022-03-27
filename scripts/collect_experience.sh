#!/usr/bin/env bash

rllib train --run=PPO --env='butia_gym.envs.manipulation.visual_grasp_env.DoRISDiverseObjectEnv' --config="{\"output\": \"../grasping-out\"}" --stop="{\"timesteps_total\": 1000000}" --framework="torch"