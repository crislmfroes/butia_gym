from setuptools import find_packages, setup

setup(
    name="butia_gym",
    packages=find_packages(),
    install_requires=[
        'stable_baselines3',
        'sb3_contrib',
        'wandb',
        'mujoco_py'
    ]
)