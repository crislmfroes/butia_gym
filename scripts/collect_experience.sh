#!/usr/bin/env bash

rllib train --run=DDPG --env='butia_gym.envs.manipulation.visual_grasp_env.DoRISDiverseObjectEnv' --config="{\"prioritized_replay\": 1, \"num_workers\": 7,  \"num_gpus\": 1, \"env_config\": {\"width\": 84, \"height\": 84},  \"output\": \"../grasping-out\"}" --stop="{\"timesteps_total\": 1000000}" --framework="torch"