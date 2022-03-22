import gym
import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet_data
from gym import spaces

class DoRISDiverseObjectEnv(KukaDiverseObjectEnv):
    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=80, isEnableSelfCollision=True, renders=False, isDiscrete=False, maxSteps=8, dv=0.06, removeHeightHack=False, blockRandom=0.3, cameraRandom=0, width=48, height=48, numObjects=5, isTest=False, **kwargs):
        super().__init__(urdfRoot, actionRepeat, isEnableSelfCollision, renders, isDiscrete, maxSteps, dv, removeHeightHack, blockRandom, cameraRandom, width, height, numObjects, isTest)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_space.shape, dtype=np.uint8)