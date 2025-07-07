import cv2
import os
import argparse
import numpy as np
import torch
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils import transform_utils
from numpy.array_api import uint8
import time
from act.policy import ACTPolicy
import pickle
from real_robot_scripts.real_robot_utils import RealRobotObsProcessor
import multiprocessing
from rebar_scripts.monitor_robot_control import monitor
import copy
from arm_control_base.utils.util_transform import convert_pos_axis_angle_singularity, convert_pose_rep
from scipy.spatial.transform import Rotation
# from multiprocessing import Process, Queue
# from multiprocessing.connection import Client
from xarm.wrapper import XArmAPI

import hydra
from omegaconf import DictConfig
# from hydra.utils import instantiate
from cam_base.camera_redis_interface import CameraRedisSubInterface

class ActPolicy:
    def __init__(
        self,
        policy_config: DictConfig,
        obs_cfg: DictConfig,
        interface_cfg: DictConfig,
        target: int,
        prim: int,
        device_id: int,
    ):
        # everything is a real object now, not a string
        # self.policy_config = policy_config
        self.checkpoint_dir = policy_config.checkpoint_dir
        self.normalize_dir  = policy_config.normalize_dir
        self.obs_cfg        = obs_cfg
        self.interface_cfg  = interface_cfg
        self.target         = target
        self.prim           = prim
        self.device_id      = device_id

        self.query_frequency = policy_config['num_queries']
        self.policy = ACTPolicy(policy_config)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {self.checkpoint_dir}')

        self.max_steps = self.interface_cfg.max_steps
        self.crop_option = self.interface_cfg.crop_option
        self.crop_list = self.interface_cfg.crop_list 
        
        self.target = torch.tensor(np.array([target])).cuda(0)
        self.prim = torch.tensor(np.array([prim])).cuda(0)
        

        loading_status = self.policy.load_state_dict(torch.load(self.checkpoint_dir))
        print(loading_status)
        with open(self.normalize_dir, 'rb') as f:
            self.stats = pickle.load(f)
        assert len(self.policy_config["prim_range"]) == 1
        self.stats = self.stats[self.policy_config["prim_range"][0]]  # self.stats["1_insert"]
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        
        # We abstract away many details in processing camera images in the class RealRobotObsProcessor. More details please refer to the class.
        self.obs_processor = RealRobotObsProcessor(self.obs_cfg, processor_name="ImageProcessor")

        # Initialize the robot
        self.arm = XArmAPI(self.interface_cfg['ip'],)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)


        # camera inteface
        self.camera_ids = self.obs_cfg.camera_ids
        self.cam_interfaces = {}
        for idx, camera_id in enumerate(self.camera_ids):
            camera_info = {"camera_id": camera_id, "camera_name": self.camera_names[idx]}
            cam_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False)
            cam_interface.start()
            self.cam_interfaces[camera_id] = cam_interface
            # self.obs_action_data[f"camera_{camera_id}"] = []

        # Initialize the FT sensor
        self.FT_option = self.interface_cfg('FT_option')
        if self.FT_option:
            self.arm.ft_sensor_enable(1)
            self.arm.ft_sensor_set_zero()
            time.sleep(0.2)
            self.arm.ft_sensor_app_set(1)
            self.arm.set_state(0)
        time.sleep(0.5)

        
    def run(self):
        count = 0
        gripper_state = np.array([1]) # -1 for open, 1 for close  #TODO: Tao: HARDCORD here
        with torch.inference_mode():
            while count < self.max_steps:
                policy_start_time = time.time()

                """robot state"""
                ee_state = self.arm.get_position_aa(is_radian=True)[1] # axis angles
                ee_state[:, :3] /= 1000.0 # Convert translation from mm → m
                # avoid singularity
                ee_state = convert_pos_axis_angle_singularity(ee_state)
                ee_gripper_state = np.concatenate((ee_state, gripper_state.reshape(gripper_state.shape[0], 1)), axis=1)
                assert self.policy_config['qpos_horizon'] == 1, "We only implement qpos_horizon = 1"
                ee_gripper_state_curr = ee_gripper_state

                assert self.query_frequency == 50, "We assume implement query_frequency = 50"
                if count % self.query_frequency == 0:
                    action_ref = ee_gripper_state_curr

                ee_gripper_state_curr_rel = convert_pose_rep(ee_gripper_state_curr, ref_pose=ee_gripper_state_curr[-1, :6], type="relative")
                import pdb; pdb.set_trace()

                """FT"""
                # TODO: FT sensor need to do normalization, if we only consider the last frame, the policy may easilly be affect by the dynamics
                if self.FT_option:
                    code, _ = self.arm.get_ft_sensor_data()
                    if code == 0:
                        force_torque = self.arm.ft_ext_force
                    else:
                        raise Exception(f"Failed to get FT sensor data: {code}")

                qpos = self.pre_process(ee_gripper_state_curr_rel)
                qpos = torch.from_numpy(qpos).float().cuda()
                qpos = qpos.unsqueeze(0) # (0, horizon, qpos_states)

                if count % self.query_frequency == 0:
                    """visual obs"""
                    curr_images = []
                    # for j in range(len(self.obs_cfg.camera_ids)):
                    for cam_id in self.camera_ids:
                        imgs_array = self.cam_interfaces[cam_id].get_img()
                        img_bgr = cv2.cvtColor(imgs_array["color"], cv2.COLOR_RGB2BGR)

                        # TODO: we need to crop the images here
                        # curr_image = cv2.resize(curr_image, (480, 320)) # 

                        if img_bgr.ndim == 3 and img_bgr.shape[-1] == 3: # Convert from (H, W, C) to (C, H, W) as required by PyTorch
                            img_bgr = img_bgr.transpose(2, 0, 1)

                            # Display the image in an OpenCV window
                            # curr_image_display = curr_image.transpose(1, 2, 0)  # Convert back to (H, W, C)
                            # cv2.imshow(f"Camera View {j}", curr_image_display)
                            # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                            #     break
                        
                        img_bgr = torch.from_numpy(img_bgr / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)  # torch.Size([1, 1, 3, 480, 640])
                        curr_images.append(img_bgr)

                    curr_images = torch.cat(curr_images, dim=1)  # torch.Size([1, 2, 3, h, w]) # should be BGR
                    torch.cuda.synchronize(); start_time = time.time()

                    all_actions = self.policy(qpos=qpos, image=curr_images, force_torque=force_torque, prim=self.prim, target=self.target) # 1 x 100 x 7   qpos, image (should be BGR), prim, target
                    action[:,:3] *= 1000.0 # Convert translation from m → mm

                    # for debugging:
                    # save_curr_images(count, curr_images)
                    torch.cuda.synchronize(); end_time = time.time()
                    print(f"inference time: {end_time - start_time}") # inference speed 3080ti 0.045s


                raw_action = all_actions[:, count % self.query_frequency]  # (1, 7)
                raw_action = raw_action.squeeze(0).cpu().to(torch.float32).numpy()  # (7,)
                action = self.post_process(raw_action)  # target_qpo

                # we need to use relative pose in terms of the initial pose
                action = convert_pose_rep(poses=action, ref_pose=action_ref, type="relative", backward=True)
                target_action = np.zeros(7)
                target_action[6] = action[6]
                target_action[:6] = convert_pos_axis_angle_singularity(action[:6])
                print(f"Action: {target_action}")


                self.arm.set_servo_cartesian_aa(target_action, speed=20, mvacc=200, is_radian=True) # action, is absolute pose list [x, y, z, rx, ry, rz] axis angles in radian

                count += 1

                policy_end_time = time.time()
                print(f"policy runtime: {policy_end_time - policy_start_time}")

                # TODO: implement the early end
                # self.close()

    def close(self):
        if self.FT_option:
            self.arm.ft_sensor_app_set(0)
            self.arm.ft_sensor_enable(0)
        self.arm.disconnect()

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):

    # Option A – manual construction
    policy = ActPolicy(
        policy_config=cfg.policy_config,
        obs_cfg=cfg.obs_cfg,
        interface_cfg=cfg.interface_cfg,
        target=cfg.target,
        prim=cfg.prim,
        device_id=cfg.device_id,
    )

    # Option B – one-liner autoinstantiation (requires _target_ in YAML)
    # policy = instantiate(cfg.policy)

    policy.run()

if __name__ == "__main__":
    main()