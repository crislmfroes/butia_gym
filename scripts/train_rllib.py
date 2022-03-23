from gc import callbacks
from pyrsistent import dq
import torch
from butia_gym.envs.manipulation.visual_grasp_env import DoRISDiverseObjectEnvWithCurriculum
#import butia_gym.envs.manipulation
#from butia_gym.envs.manipulation.pick_and_place_env import DoRISPickAndPlaceEnv
#from butia_gym.envs.manipulation.pick_and_place_task import DoRISPickAndPlaceTask
from ray.rllib.agents.callbacks import DefaultCallbacks#, MultiCallbacks
from ray.rllib.agents import sac, es, dreamer, dqn, ddpg, ppo
from ray.rllib import *
from ray.tune.integration.wandb import WandbLoggerCallback
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

def curriculum_fn(
    train_results, task_settable_env, env_ctx
):
    """Function returning a possibly new task to set `task_settable_env` to.
    Args:
        train_results (dict): The train results returned by Trainer.train().
        task_settable_env (TaskSettableEnv): A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx (EnvContext): The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.
    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    new_task = int(np.log10(train_results["episode_reward_mean"]) + 2.1)
    # Clamp between valid values, just in case:
    new_task = max(min(new_task, 5), 1)
    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task

if __name__ == '__main__':
    wandb.login()
    NUM_CPUS=8
    ray.init(num_cpus=NUM_CPUS, num_gpus=1)
    #env_name = 'butia_gym.envs.manipulation.grasp_env.DoRISGraspEnv'
    env_name = 'butia_gym.envs.manipulation.visual_grasp_env.DoRISDiverseObjectEnv'
    #tune.register_env(env_name, lambda cfg: gym.make(env_name))
    config = sac.DEFAULT_CONFIG.copy()
    config['framework'] = 'torch'
    #config['num_gpus'] = 1.0/NUM_CPUS
    #config['num_gpus'] = 1
    #config['num_workers'] = 0
    #config['clip_actions'] = False
    #config['num_workers'] = 0
    #config['num_gpus_per_worker'] = 1.0/NUM_CPUS
    #config['num_gpus_per_worker'] = 1
    #config['compress_observations'] = True
    #config['num_envs_per_worker'] = int(96/NUM_CPUS)
    #config['training_intensity'] = 1
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
    config['env_config']['isDiscrete'] = False
    config['env_config']['width'] = 42
    config['env_config']['height'] = 42
    #config['model']['dim'] = 42
    '''config['model']['conv_filters'] = [
        [16,8,4],
        [32,4,2],
        [64,11,1]
    ]'''
    #config['env_task_fn'] = curriculum_fn
    #config['env_config']['start_level'] = 1
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
    '''config['exploration_config'] = {
        "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 288,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }'''
    config['prioritized_replay'] = True
    config['env'] = env_name
    callbacks = [WandbLoggerCallback('kuka-manipulation', 'DRL')]
    tune.run(
        #ppo.PPOTrainer,
        #dqn.DQNTrainer,
        sac.SACTrainer,
        #es.ESTrainer,
        #dreamer.DREAMERTrainer,
        #ddpg.ApexDDPGTrainer,
        checkpoint_freq=1,
        config=config,
        callbacks=callbacks,
        stop={
            "training_iteration": 10000
        },
        reuse_actors=True,
        #resume=True,
    )
