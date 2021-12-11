from stable_baselines3.common import monitor
import butia_gym
import gym
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.tqc import tqc
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper
from stable_baselines3.common.buffers import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.her.her_replay_buffer import *
import yaml
import wandb

if __name__ == '__main__':
    with open('configs/pick_place_tqc_her.yaml', 'r') as f:
        config = yaml.load(f)
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True
    )
    if 'replay_buffer_class' in config:
        config['replay_buffer_class'] = eval(config['replay_buffer_class'])
    def make_env():
        env = gym.make('DoRISPickAndPlace-v1')
        env = TimeFeatureWrapper(env)
        env = Monitor(env)
        return env
    env = DummyVecEnv([make_env,])
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    model = tqc.TQC(**config, env=env, verbose=1, tensorboard_log=f"runs/{run.id}")
    result = model.learn(1000000, callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2
    ))
    run.finish()
