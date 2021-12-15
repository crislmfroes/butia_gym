#!/usr/bin/bash

rllib train --run SAC --env 'butia_gym.envs.manipulation.pick_and_place.DoRISPickAndPlaceEnv' --config '{"horizon": 50, "env_config": {"reward_type": "dense"}, "num_gpus": 1}' --torch