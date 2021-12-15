import butia_gym
import gym
from pyglet.window import key
import numpy as np

a = np.array([0.3, 0.0, 0.0, 0.0])
env = butia_gym.envs.DoRISPickAndPlaceEnv()
#env = gym.envs.robotics.FetchPickAndPlaceEnv()
env.render()
while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        s, r, done, info = env.step(a)
        if steps <= 100:
            if s['observation'][2] < 0.8:
                a[2] = 0.8
            else:
                a[2] = 0.0
        if steps > 100 and steps <= 200:
            a[:3] = s['observation'][6:9]
            #if abs(s['observation'][0] - s['achieved_goal'][0]) < 0.1 and abs(s['observation'][1] - s['achieved_goal'][1]) < 0.1:
            a[3] = 1.0
        if steps > 200 and steps <= 300:
            a[3] = -1.0
        if steps > 300:
            a[:3] = (s['desired_goal'] - s['achieved_goal'] + s['observation'][6:9])
        if steps % 100 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        env.render()
        if done or restart: break
env.close()