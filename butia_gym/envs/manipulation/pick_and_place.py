import os
from gym import utils
from butia_gym.envs.manipulation.doris_manipulation_env import DoRISManipulationEnv


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "doris", "pick_and_place.xml"))


class DoRISPickAndPlaceEnv(DoRISManipulationEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "dorso_arm_base": 0.2,
            "forearm_shoulder_joint": 0.75,
            "roll_joint": -1.5,
            "yaw_joint": 1.5,
            "elbow_joint": -1.6,
        }
        DoRISManipulationEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.1,
            target_range=0.1,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)