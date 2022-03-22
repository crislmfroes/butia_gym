from gc import callbacks
import torch
#import butia_gym.envs.manipulation
#from butia_gym.envs.manipulation.pick_and_place_env import DoRISPickAndPlaceEnv
#from butia_gym.envs.manipulation.pick_and_place_task import DoRISPickAndPlaceTask
from ray.rllib.agents.callbacks import DefaultCallbacks#, MultiCallbacks
from ray.rllib.agents import sac, es, dreamer
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

class HerCallback(DefaultCallbacks):
    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        '''
        Test of Hindsight experience Replay with the optimum goal and a random goal. '''
        HER_OPTIMUM = policy.config['env_config']['HER_OPT']
        if policy.config['env_config']['HER_RAND_GOALS']>0:
            HER_RANDOM = True
        else:
            HER_RANDOM = False
        RAND_GOALS = policy.config['env_config']['HER_RAND_GOALS']
        print(f'Start length: {len(train_batch)}')

        def reward(goal, hit):
            return DoRISPickAndPlaceTask.reward_function(hit, goal, 0.05)

        DESIRED_GOAL_START = -3
        DESIRED_GOAL_END = None
        ACHIEVED_GOAL_START = -6
        ACHIEVED_GOAL_END = -3

        if HER_OPTIMUM:
            train_batch_her_opt = copy.deepcopy(train_batch)
            for i in range(len(train_batch_her_opt['obs'])):
                #train_batch_her_opt['obs'][i][0] = scaling(train_batch_her_opt['obs'][i]['achieved_goal'])
                train_batch_her_opt['obs'][i][DESIRED_GOAL_START:DESIRED_GOAL_END] = train_batch_her_opt['obs'][i][ACHIEVED_GOAL_START:ACHIEVED_GOAL_END]
                #train_batch_her_opt['infos'][i]['diff_to_goal'] = 0.0
                #train_batch_her_opt['new_obs'][i][0] = scaling(train_batch_her_opt['obs'][i]['achieved_goal'])
                train_batch_her_opt['rewards'][i] = reward(train_batch_her_opt['obs'][i][ACHIEVED_GOAL_START:ACHIEVED_GOAL_END], train_batch_her_opt['obs'][i][DESIRED_GOAL_START:DESIRED_GOAL_END])

        if HER_RANDOM:
            FIRST_RAND = True
            for rand in range(RAND_GOALS):
                train_batch_her_rand = copy.deepcopy(train_batch)
                GOAL_IDX = random.choice(range(len(train_batch_her_rand['obs'])))
                RAND_GOAL = train_batch_her_rand['obs'][GOAL_IDX][ACHIEVED_GOAL_START:ACHIEVED_GOAL_END]
                for i in range(len(train_batch_her_rand)):
                    #train_batch_her_rand['obs'][i][0] = scaling(RAND_GOAL)
                    train_batch_her_rand['obs'][i][DESIRED_GOAL_START:DESIRED_GOAL_END] = RAND_GOAL
                    #train_batch_her_rand['obs'][i]['diff_to_goal'] = abs(RAND_GOAL-train_batch_her_rand['infos'][i]['achieved_goal'])
                    #train_batch_her_rand['new_obs'][i][0] = train_batch_her_rand['obs'][i][0]
                    train_batch_her_rand['rewards'][i] = reward(RAND_GOAL, train_batch_her_rand['obs'][i][ACHIEVED_GOAL_START:ACHIEVED_GOAL_END])
                if FIRST_RAND:
                    train_batch_her_rand_comb = copy.deepcopy(train_batch_her_rand)
                    FIRST_RAND = False
                else:
                    pass
                    #train_batch_her_rand_comb = SampleBatch.concat_samples([train_batch_her_rand_comb, train_batch_her_rand])

        if HER_RANDOM and HER_OPTIMUM:
            train_batch = SampleBatch.concat_samples([train_batch, train_batch_her_opt, train_batch_her_rand_comb])
        elif HER_RANDOM:
            train_batch = SampleBatch.concat_samples([train_batch, train_batch_her_rand_comb])
        elif HER_OPTIMUM:
            train_batch = SampleBatch.concat_samples([train_batch, train_batch_her_opt])

if __name__ == '__main__':
    wandb.login()
    #ray.init(num_cpus=9, num_gpus=1)
    #env_name = 'butia_gym.envs.manipulation.grasp_env.DoRISGraspEnv'
    env_name = 'pybullet_envs.bullet.KukaDiverseObjectEnv'
    #tune.register_env(env_name, lambda cfg: gym.make(env_name))
    config = sac.DEFAULT_CONFIG.copy()
    config['framework'] = 'torch'
    config['num_gpus'] = 1.0/8.0
    config['num_workers'] = 1
    config['num_gpus_per_worker'] = 1.0/8.0
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
    tune.run(
        #sac.SACTrainer,
        #es.ESTrainer,
        dreamer.DREAMERTrainer,
        checkpoint_freq=1,
        config=config,
        #callbacks=callbacks,
        stop={
            "training_iteration": 1000000
        },
        #resume=True,
    )
