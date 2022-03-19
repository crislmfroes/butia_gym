from butia_gym.envs.manipulation.pick_and_place_task import DoRISPickAndPlaceTask
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from butia_gym.envs.manipulation.doris_robot import DoRISRobot
import numpy as np
import time

class DoRISPickAndPlaceEnv(RobotTaskEnv):
    def __init__(self, render: bool = False, reward_type: str = "sparse", **kwargs):
        self.sim = PyBullet(render=render)
        self.robot = DoRISRobot(self.sim)
        self.task = DoRISPickAndPlaceTask(self.sim, reward_type=reward_type, get_ee_position=self.robot.get_ee_position)
        RobotTaskEnv.__init__(self)
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
        return obs