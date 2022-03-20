from sb3_contrib.tqc.tqc import TQC
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper
from butia_gym.envs.manipulation.pick_and_place_env import DoRISPickAndPlaceEnv
import gym
import numpy as np


if __name__ == '__main__':
    env = DoRISPickAndPlaceEnv(render=True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    env = TimeFeatureWrapper(env, max_steps=50, test_mode=False)
    agent = TQC.load('agents/PandaPickAndPlace-v1.zip', env=env, device='cpu', custom_objects={
        'learning_rate': 1e-4,
        'replay_buffer_kwargs': {
            'max_episode_length': 50
        }
    })
    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)[0]
            obs, reward, done, info = env.step(action)
            env.render('human')