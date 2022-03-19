from butia_gym.envs.manipulation.pick_and_place_task import DoRISPickAndPlaceTask
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from butia_gym.envs.manipulation.doris_robot import DoRISRobot
import numpy as np
import time
from gym import spaces

class DoRISPickAndPlaceEnv(RobotTaskEnv):
    def __init__(self, render: bool = False, reward_type: str = "sparse", **kwargs):
        self.sim = PyBullet(render=render)
        self.robot = DoRISRobot(self.sim)
        self.task = DoRISPickAndPlaceTask(self.sim, reward_type=reward_type, get_ee_position=self.robot.get_ee_position)
        self.seed()  # required for init for can be changer later
        obs = self.reset()
        observation_shape = obs.shape
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=observation_shape)
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        '''self.observation_space['observation'].low = -50*np.ones(shape=self.observation_space['observation'].shape)
        self.observation_space['observation'].high = 50*np.ones(shape=self.observation_space['observation'].shape)
        self.observation_space['desired_goal'].low = -50*np.ones(shape=self.observation_space['desired_goal'].shape)
        self.observation_space['desired_goal'].high = 50*np.ones(shape=self.observation_space['desired_goal'].shape)
        self.observation_space['achieved_goal'].low = -50*np.ones(shape=self.observation_space['achieved_goal'].shape)
        self.observation_space['achieved_goal'].high = 50*np.ones(shape=self.observation_space['achieved_goal'].shape)'''
    
    def _get_obs(self):
        obs = super()._get_obs()
        if self.observation_space is not None:
            for k in obs.keys():
                obs[k] = np.clip(obs[k], self.observation_space[k].low, self.observation_space[k].high)
        return np.concatenate([v for k, v in obs.items()])