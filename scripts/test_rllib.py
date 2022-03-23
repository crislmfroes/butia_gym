from gc import callbacks
import torch
#import butia_gym.envs.manipulation
from butia_gym.envs.manipulation.visual_grasp_env import DoRISDiverseObjectEnv
#from butia_gym.envs.manipulation.pick_and_place_env import DoRISPickAndPlaceEnv
#from butia_gym.envs.manipulation.pick_and_place_task import DoRISPickAndPlaceTask
from ray.rllib.agents.callbacks import DefaultCallbacks#, MultiCallbacks
from ray.rllib.agents import sac, es, dreamer, dqn, ddpg
from ray.rllib import *
#from ray.tune.integration.wandb import WandbLoggerCallback
import ray
from ray import tune
import copy
import random
import wandb
import gym
from gym.spaces import Box, Dict
import numpy as np

if __name__ == '__main__':
    #ray.init(num_cpus=9, num_gpus=1)
    #env_name = 'butia_gym.envs.manipulation.grasp_env.DoRISGraspEnv'
    env_name = 'butia_gym.envs.manipulation.visual_grasp_env.DoRISDiverseObjectEnv'
    #tune.register_env(env_name, lambda cfg: gym.make(env_name))
    config = es.DEFAULT_CONFIG.copy()
    config['framework'] = 'torch'
    #config['num_gpus'] = 1
    #config['num_gpus_per_worker'] = 1
    #config['num_gpus_per_worker'] = 1
    #config['num_gpus_per_trial'] = 1
    #config['horizon'] = 8
    #config['model']['dim'] = 48
    #config['disable_env_checking'] = True
    #config['evaluation_config']['env_config']['render'] = True
    #config['env_config']['reward_threshold'] = 5.0
    #config['env_config']['render'] = True
    config['env_config']['renders'] = False
    config['env_config']['width'] = 42
    config['env_config']['height'] = 42
    #config['env_config']['frame_skip'] = 1
    #config['env_config']['HER_RANDOM'] = True
    #config['env_config']['HER_OPT'] = True
    #config['env_config']['clip_obs'] = True
    #config['env_config']['HER_RAND_GOALS'] = 4
    #config['env_config']['max_steps'] = 50
    #config['env_config']['range_goal'] = 50
    #config['callbacks'] = MultiCallbacks([
    #    HerCallback,
    #])
    config['env'] = env_name
    #callbacks = [WandbLoggerCallback('kuka-manipulation', 'DRL')]
    agent = es.ESTrainer(config=config, env=env_name)
    agent.restore('../ray_results/ESTrainer/ESTrainer_butia_gym.envs.manipulation.visual_grasp_env.DoRISDiverseObjectEnv_a007a_00000_0_2022-03-22_11-16-11/checkpoint_24')
    env = DoRISDiverseObjectEnv(width=42, height=42, renders=True)
    for i in range(10):
        done = False
        obs = env.reset()
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)