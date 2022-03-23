import random
from typing import Any, Dict, List, Tuple
import gym
import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet_data
import pybullet as p
from gym import spaces
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

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
        return reward

class DoRISDiverseObjectEnvWithCurriculum(TaskSettableEnv):

    RANGE_LEVELS = [0.05, 0.1, 0.2, 0.3]

    def __init__(self, config: EnvContext) -> None:
        self.config = config
        self.grasp_env = None
        self._make_grasp_env()
        self.cur_level = config.get('start_level', 1)
        self.observation_space = self.grasp_env.observation_space
        self.action_space = self.grasp_env.action_space
        self.switch_env = False

    def reset(self) -> Any:
        if self.switch_env:
            self.switch_env = False
            self._make_grasp_env()
        return self.grasp_env.reset()

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.grasp_env.step(action)
        reward *= 10 ** (self.cur_level - 1)
        return obs, reward, done, info

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks: int) -> List[Any]:
        return [random.randint(1, 4) for _ in range(n_tasks)]
    
    @override(TaskSettableEnv)
    def get_task(self) -> Any:
        return self.cur_level

    @override(TaskSettableEnv)
    def set_task(self, task: Any) -> None:
        self.cur_level = task
        self.switch_env = True

    def _make_grasp_env(self):
        self.grasp_env = DoRISDiverseObjectEnv(blockRandom=self.RANGE_LEVELS[self.cur_level - 1], width=42, height=42)
