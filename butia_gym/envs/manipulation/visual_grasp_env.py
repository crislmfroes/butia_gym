import gym
import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet_data
import pybullet as p
from gym import spaces

class DoRISDiverseObjectEnv(KukaDiverseObjectEnv):
    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=80, isEnableSelfCollision=True, renders=False, isDiscrete=False, maxSteps=8, dv=0.06, removeHeightHack=False, blockRandom=0.3, cameraRandom=0, width=48, height=48, numObjects=5, isTest=False, **kwargs):
        super().__init__(urdfRoot, actionRepeat, isEnableSelfCollision, renders, isDiscrete, maxSteps, dv, removeHeightHack, blockRandom, cameraRandom, width, height, numObjects, isTest)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_space.shape, dtype=np.uint8)

    def _reward(self):
        """Calculates the reward for the episode.
        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        self._graspSuccess = 0
        distances = np.zeros(shape=(len(self._objectUids),))
        for i, uid in enumerate(self._objectUids):
            pos, _ = p.getBasePositionAndOrientation(uid)
            ee_pos = self._kuka.endEffectorPos
            distances[i] = np.linalg.norm(np.array(pos) - np.array(ee_pos))
            # If any block is above height, provide reward.
            if pos[2] > 0.2:
                self._graspSuccess += 1
                reward += 100
                break
        min_dist = np.min(distances)
        reward += -min_dist
        return 