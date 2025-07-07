import cv2
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
import threading
from rebar_scripts.util_processing import safety_filter
from rebar_scripts.util_transform import convert_pos_axis_angle_singularity, mat2pos_axis
from scipy.spatial.transform import Rotation
from multiprocessing import Process, Queue
from multiprocessing.connection import Client
from sam2_ulti import Crop_and_Msk_Obs
import json, os, re
import torch.nn.functional as F

def record_video(save_dir):
    # Open the video capture (adjust frame width/height as needed)
    cap = cv2.VideoCapture('/dev/video10')
    if not cap.isOpened():
        print("Error: Could not open /dev/video10")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the filename by checking the save directory
    video_index = 0
    while os.path.exists(os.path.join(save_dir, f"{video_index}.mp4")):
        video_index += 1
    video_filename = os.path.join(save_dir, f"{video_index}.mp4")
    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0  # Adjust FPS as needed
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    print(f"Recording video to {video_filename}. Press 'q' to stop recording.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        out.write(frame) # Write the frame to the video file
        cv2.imshow("Video Recording", frame) # Display the frame (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video recording...")
            break
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def convert_pose_rep(poses, ref_pose, type="relative", backward=False):
    """
    poses,   # n x 6 array
    ref_pose, # 1 x 6 array
    type, # "relative"
    return: n x 6 array, relative xyz and 3 axis angels in terms of ref_pose
    """

    if not backward:
        if type == "relative":
            initial_xyz = ref_pose[:3]
            initial_axis_angle = ref_pose[3:6]
            initial_quat = Rotation.from_rotvec(initial_axis_angle)
            relative_pose = poses.copy()
            for i in range(poses.shape[0]):
                # xyz
                relative_pose[i, :3] = initial_quat.inv().apply(
                    (relative_pose[i, :3] - initial_xyz))  # TODO: Unify all the functions
                # axis angles
                rot_quat_i = Rotation.from_rotvec(relative_pose[i, 3:6])
                quat_diff = initial_quat.inv() * rot_quat_i
                relative_pose[i, 3:6] = quat_diff.as_rotvec()
        else:
            raise NotImplementedError
        return relative_pose
    else:
        if len(poses.shape) != 1:
            if type == "relative":
                initial_xyz = ref_pose[:3]
                initial_axis_angle = ref_pose[3:6]
                initial_quat = Rotation.from_rotvec(initial_axis_angle)

                abs_pose = np.zeros((poses.shape[0], 6))
                for i in range(poses.shape[0]):
                    # xyz
                    abs_pose[i, :3] = (initial_quat.apply(poses[i, :3])) + initial_xyz
                    # axis angles
                    rot_quat_i = Rotation.from_rotvec(poses[i, 3:6])
                    abs_quat = initial_quat * rot_quat_i
                    abs_pose[i, 3:6] = abs_quat.as_rotvec()
            else:
                raise NotImplementedError
        else:
            if type == "relative":
                initial_xyz = ref_pose[:3]
                initial_axis_angle = ref_pose[3:6]
                initial_quat = Rotation.from_rotvec(initial_axis_angle)

                abs_pose = poses.copy()
                # xyz
                abs_pose[:3] = (initial_quat.apply(poses[:3])) + initial_xyz
                # axis angles
                rot_quat_i = Rotation.from_rotvec(poses[3:6])
                abs_quat = initial_quat * rot_quat_i
                abs_pose[3:6] = abs_quat.as_rotvec()
            else:
                raise NotImplementedError
        return abs_pose

def monitor_state_process(mat, initial_ee_pose):
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    ee_data_new = np.zeros(6)

    mat = mat.reshape(4, 4, order='F')
    # To avoid singularity of axis angle representation
    rot_mat = mat[:3, :3]
    rot_mat = A[:3, :3] @ rot_mat
    trans_vec = mat[:3, 3]
    rot = Rotation.from_matrix(rot_mat)
    rot_vec = rot.as_rotvec()  # axis angle
    ee_data_new[:3] = trans_vec
    ee_data_new[3:] = rot_vec

    ee_data_new_relative = np.zeros(6)
    ee_data_new_relative[:3] = ee_data_new[:3] - initial_ee_pose[0,:3]
    current_quat = Rotation.from_rotvec(ee_data_new[3:])
    initial_quat = Rotation.from_rotvec(initial_ee_pose[0,3:6])
    quat_diff = current_quat * initial_quat.inv()
    ee_data_new_relative[3:6] = quat_diff.as_rotvec()

    return ee_data_new_relative, rot_mat

def monitor_action_process(target_action, initial_ee_pose):
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    # print(f"before target_action: {target_action}")
    rot2 = Rotation.from_rotvec(target_action[3:6])
    rot_mat2 = rot2.as_matrix()
    rot_mat2 = A[:3,:3]  @ rot_mat2  # A.T
    # print(rot_mat2)
    rot_new2 = Rotation.from_matrix(rot_mat2)
    rot_vec_new2 = rot_new2.as_rotvec()  # axis angle
    target_action[3:6] = rot_vec_new2
    # print(f"action: {target_action}")

    target_action_relative = np.zeros(6)
    target_action_relative[:3] = target_action[:3] - initial_ee_pose[0,:3]
    current_quat = Rotation.from_rotvec(target_action[3:])
    initial_quat = Rotation.from_rotvec(initial_ee_pose[0,3:6])
    quat_diff = current_quat * initial_quat.inv()
    target_action_relative[3:6] = quat_diff.as_rotvec()

    return target_action_relative, rot_mat2

def send_images_to_server(address, result_queue, camera_names, rebar_type, obs, process_start,count, slot_name):
    """Child process that sends image data to the inference server."""
    with Client(('localhost', address), authkey=b'secret') as conn:
        message = {"camera_names": camera_names, "rebar_type": rebar_type, "obs": obs, "process_start": process_start,"count": count, "slot_name": slot_name}
        conn.send(message)
        result = conn.recv()
        result_queue.put(result)

def save_curr_images(count,curr_images, save_path):
    # torch.Size([1, 2, 3, h, w]) # should be BGR
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    b, img_num, c, img_h, img_w = curr_images.shape
    start = time.time()

    if img_num > 480 or img_h > 320:
        # if too big, resize down to exactly 480×320
        imgs = curr_images.view(b * img_num, c, img_h, img_w)  # → [2, 3, 900, 960]
        resized = F.interpolate(
            imgs,
            size=(300, 320),  # (H, W)
            mode='bilinear',  # or 'nearest'
            align_corners=False
        )  # → [2, 3, 300, 320]
        # 3) Restore original [batch, num_images] structure
        curr_images = resized.view(b, img_num, c, 300, 320)  # → [1, 2, 3, 300, 320]

    for i in range(img_num):
        img = curr_images[0,i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0)) # Convert from CHW to HWC
        if img.dtype == np.float32 or img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        # h, w = img.shape[:2]
        # if w > 480 or h > 320:
        #     img = cv2.resize(img, (320, 300), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{save_path}/{count}_{i}.jpg", img)
    print(f"debug save img time: {time.time() - start}")

class Act_policy():
    def __init__(self, 
                 checkpoint_dir="act/ckpt_trained/insert_v9/policy_last.ckpt",
                 normalize_dir ="act/ckpt_trained/insert_v9/dataset_stats.pkl",
                 monitor_option = False,
                 device_id = 0, # "cuda:" + str(args.device_id)
                 max_steps = 600,
                 video = None, # "None or a folder path, /home/autostruct-1/Dropbox/insert7",
                 obs_cfg_path = "rebar_configs/real_robot_observation_cfg.yml",
                 rebar_type = "rebar1", #  "rebar1"  "rebar2",
                 target = 0,
                 prim = 0,
                 slot_name="slot_4",
                 crop_option=True,
                 crop_list = [[480, 0, 1440, 900], [480, 0, 1440, 900]],
                 policy_config = None,
                 interface_cfg="rebar_configs/franka_interface.yml",
                 controller_cfg_path = "rebar_configs/osc-pose-controller.yml",
                 controller_type="OSC_POSE",
                 sam_option = True,
                 record_trajectory=False,
                 policy_name ="",
                 early_stop = True,
                 state_action_type = "relative",
            ):
                 
        self.checkpoint_dir = checkpoint_dir
        self.normalize_dir = normalize_dir
        self.monitor = monitor_option
        self.device_id = device_id
        self.max_steps = max_steps
        self.video = video
        self.obs_cfg_path = obs_cfg_path
        self.obs_cfg = YamlConfig(self.obs_cfg_path).as_easydict()
        self.state_dim = 7
        self.rebar_type = rebar_type 
        self.target = torch.tensor(np.array([target])).cuda(0)
        self.prim = torch.tensor(np.array([prim])).cuda(0)
        self.crop_option = crop_option
        self.crop_list = crop_list 
        self.policy_config = policy_config
        self.query_frequency = self.policy_config['num_queries']
        self.slot_name = slot_name
        self.policy = ACTPolicy(self.policy_config)
        self.sam_option = sam_option
        self.record_trajectory = record_trajectory
        self.policy_name = policy_name
        self.early_stop = early_stop
        self.state_action_type = state_action_type

        loading_status = self.policy.load_state_dict(torch.load(self.checkpoint_dir))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {self.checkpoint_dir}')
        with open(self.normalize_dir, 'rb') as f:
            self.stats = pickle.load(f)
        assert len(self.policy_config["prim_range"]) == 1
        self.stats = self.stats[self.policy_config["prim_range"][0]]  # self.stats["1_insert"]
        
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
    
        self.qpos_history = np.zeros((8000, self.state_dim))
        self.state_list = []
        # We abstract away many details in processing camera images in the class RealRobotObsProcessor. More details please refer to the class.
        self.obs_processor = RealRobotObsProcessor(self.obs_cfg, processor_name="ImageProcessor")
    

        self.robot_interface = FrankaInterface(interface_cfg, automatic_gripper_reset = False)
        self.controller_cfg = YamlConfig(controller_cfg_path).as_easydict()
        self.controller_type = controller_type
        self.last_crop_masked_obs = None

        if self.monitor:
            manager = multiprocessing.Manager()  # Create a manager for shared objects
            self.times = manager.list()  # Shared list for times
            self.state_points = manager.list([manager.list() for _ in range(6)])  # Shared 6-element list for state
            self.action_points = manager.list(
                [manager.list() for _ in range(6)])  # Shared 6-element list for action  # Shared 6-element list for action
            monitor_save_path = "./last_robot_state_action_fig_act.png"
            monitor_process = multiprocessing.Process(target=monitor, args=(self.times, self.state_points, self.action_points, monitor_save_path))
            monitor_process.start()
        self.Crop_module = Crop_and_Msk_Obs()

        if self.record_trajectory:
            if not os.path.exists(self.record_trajectory):
                os.makedirs(self.record_trajectory)
            self.trajectory_data = []


    def run(self):
        self.robot_interface.reset()
        print("Deoxys is reset...")
        time.sleep(1)
        count = 0
        result_queue = Queue()

        if self.record_trajectory:
            existing = os.listdir(self.record_trajectory)
            nums = []
            for fn in existing:
                m = re.match(r"^test_(\d+)_.*\.json$", fn)
                # m = re.match(r'test_(\d+)_(good|poor)\.json$', fn)
                if m: nums.append(int(m.group(1)))
            next_num = max(nums, default=0) + 1

        if self.video is not None:
            if not os.path.exists(self.video):
                os.makedirs(self.video)
            # Start the video recording thread as a daemon thread so it stops when the main thread exits.
            video_thread = threading.Thread(target=record_video, args=(self.video,), daemon=True)
            video_thread.start()

        with torch.inference_mode():
            while count < self.max_steps:
                policy_start_time = time.time()
                if count == 0:
                    ee_mat_debug = np.asarray(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).T
                    initial_pose = mat2pos_axis(ee_mat_debug) # for monitoring only
                    print(f"initial_pose is {initial_pose} (Before A)")
                    if self.monitor or self.early_stop:
                        initial_pose = convert_pos_axis_angle_singularity(initial_pose)
                        # print(f"initial_pose is {initial_pose} (After A)")
                        initial_pose = initial_pose.reshape(1, 6)
                """online obs"""
                # Skip if robot states have never been received so far

                if "tying" in self.policy_config["prim_range"][0]:
                    if len(self.robot_interface._state_buffer) == 0:
                        print("[initialize] Waiting for robot state...")
                        continue
                else:
                    if len(self.robot_interface._state_buffer) == 0 or len(self.robot_interface._gripper_state_buffer) == 0:
                        print("[initialize] Waiting for robot state and griper state...")
                        continue

                # 2. Get the latest state of the robot (including the gripper)
                last_state = self.robot_interface._state_buffer[-1]
                if "tying" in self.policy_config["prim_range"][0]:
                    last_gripper_state = 0
                else:
                    last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
                self.obs_processor.get_real_robot_state(last_state, last_gripper_state)
                self.obs_processor.get_real_robot_img_obs(crop_option=self.crop_option, crop_list=self.crop_list)
                obs = self.obs_processor.obs

                # ------------------------- SAM2 ------------------------- #
                print(f"count: {count}")
                if self.sam_option:
                    if count == 0 or count % 10 == 8: # The base inference speed of SAM2 is 0.2 seconds, so we process two frequencies in advance.
                        # Launch the inference-sending process as a separate child process.
                        process_start = time.time()

                        # for j in range(2):
                        #     curr_image = obs[f"agentview{j}_rgb"]
                        #     cv2.imwrite(f"rebar_scripts/SAM2_debug_log/{count}_{j}.jpg", curr_image)

                        inference_process = Process(target=send_images_to_server,
                                                    args=(6000, result_queue, self.obs_cfg.camera_names, self.rebar_type, obs, process_start,
                                                    count,  self.slot_name),
                                                    daemon=True)
                        inference_process.start()
                # ------------------------- SAM2 ------------------------- #
                ee_data_new = np.zeros((self.policy_config['qpos_horizon'], 7))
                if count <= self.policy_config['qpos_horizon']-1: # TODO: may need to be -1
                    ee_data_new[self.policy_config['qpos_horizon']-count:, :] = self.qpos_history[:count, :]
                    ee_data_new[:self.policy_config['qpos_horizon']-count-1, :] = np.zeros(7)
                else:
                    ee_data_new[:-1,:] = self.qpos_history[count-self.policy_config['qpos_horizon']+1:count,:]

                array = obs["ee_states"]
                mat = array.reshape(4, 4, order='F').copy()
                # To avoid singularity
                ee_data_curr = mat2pos_axis(mat)
                ee_data_new[-1,:6] = convert_pos_axis_angle_singularity(ee_data_curr)
                ee_data_new[-1, 6] = obs["gripper_states"]
                self.qpos_history[count, :] = ee_data_new[-1, :]
                if self.state_action_type == "relative":
                    ee_data = convert_pose_rep(ee_data_new, ref_pose=ee_data_new[-1, :6], type="relative")
                else: # "absolute"
                    ee_data = ee_data_new

                if count % self.query_frequency == 0:
                    action_ref = ee_data_new[-1, :6]

                """Policy prediction"""
                qpos = self.pre_process(ee_data)
                qpos = torch.from_numpy(qpos).float().cuda()
                qpos = qpos.unsqueeze(0) # (0, horizon, qpos_states)

                if count % self.query_frequency == 0:
                    if self.sam_option:
                        # if count == 0:
                        while result_queue.empty():
                            time.sleep(0.005)
                        if not result_queue.empty():
                            last_masks_list = result_queue.get()
                        st = time.time()

                        # for j in range(2):
                        #     curr_image = obs[f"agentview{j}_rgb"]
                        #     cv2.imwrite(f"rebar_scripts/SAM2_debug_log/{count}_{j}.jpg", curr_image)

                        curr_images = self.Crop_module.mask_obs(obs, self.obs_cfg.camera_names, last_masks_list, slot_name=self.slot_name) # TODO: we may want to improve the speed of this step
                        last_masks_list = None
                        torch.cuda.synchronize();  print(f"add mask time: {time.time() - st}")
                        curr_images = torch.cat(curr_images, dim=1)  # torch.Size([1, 2, 3, h, w]) # should be BGR
                        torch.cuda.synchronize(); start_time = time.time()
                    else:
                        curr_images = []
                        for j in range(len(self.obs_cfg.camera_ids)):
                            curr_image = obs[f"agentview{j}_rgb"] # raw image from obs is RGB!
                            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR) # raw image from obs is RGB!
                            # curr_image = cv2.resize(curr_image, (480, 320)) # TODO: we hard code the image size here. tying:(480, 320) insertion: (960, 900)
                            if curr_image.ndim == 3 and curr_image.shape[-1] == 3: # Convert from (H, W, C) to (C, H, W) as required by PyTorch
                                curr_image = curr_image.transpose(2, 0, 1)
                                # Display the image in an OpenCV window
                                # curr_image_display = curr_image.transpose(1, 2, 0)  # Convert back to (H, W, C)
                                # cv2.imshow(f"Camera View {j}", curr_image_display)
                                # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                                #     break
                            curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)  # torch.Size([1, 1, 3, 480, 640])
                            curr_images.append(curr_image)
                        curr_images = torch.cat(curr_images, dim=1)  # torch.Size([1, 2, 3, h, w]) # should be BGR
                        torch.cuda.synchronize(); start_time = time.time()

                    all_actions = self.policy(qpos = qpos, image = curr_images, prim=self.prim, target =self.target) # 1 x 100 x 7   qpos, image (should be BGR), prim, target
                    # for debugging:
                    save_curr_images(count, curr_images, save_path=os.path.join(self.record_trajectory, str(next_num)))
                    torch.cuda.synchronize(); end_time = time.time()
                    print(f"inference time: {end_time - start_time}") # inference speed 3080ti 0.045s


                raw_action = all_actions[:, count % self.query_frequency]  # (1, 7)
                raw_action = raw_action.squeeze(0).cpu().to(torch.float32).numpy()  # (7,)
                action = self.post_process(raw_action)  # target_qpo
                if self.state_action_type == "relative":
                    # we need to use relative pose in terms of the initial pose
                    action = convert_pose_rep(poses=action, ref_pose=action_ref, type="relative", backward=True)
                # action[-1] = np.where(action[-1] < -0.8, -1, 1) # make the gripper commend to be 1 or 0
                target_action = np.zeros(7)
                target_action[6] = action[6]
                target_action[:6] = convert_pos_axis_angle_singularity(action[:6])
                print(f"Action: {target_action}")
                # print(f"Action: {target_action}")
                # if not safety_filter(target_action):
                #     print("It is out of the safe range!")
                #     break
                # import pdb; pdb.set_trace()

                if self.record_trajectory:
                    # record the raw EE pose and action for this step
                    self.trajectory_data.append({
                        'ee_action': target_action.tolist(),
                        'ee_state': ee_data_curr.tolist()
                    })

                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=target_action,
                    controller_cfg=self.controller_cfg,
                ) # note: Due to singularity Action: [ 0.45555964 -0.11069027  0.17342131  3.13721652 -0.14118875  0.04813786 1.00000429] is almost the same with Action: [ 0.45789659 -0.11023864  0.17223237  3.13291941 -0.13433651  0.04722853 1.00000906]

                torch.cuda.synchronize();
                policy_end_time = time.time()
                print(f"policy runtime: {policy_end_time - policy_start_time}")

                count += 1

                if self.early_stop or self.monitor:
                    target_state_monitor = monitor_state_process(copy.deepcopy(np.array(last_state.O_T_EE)), initial_pose)[0]
                    target_action_monitor = monitor_action_process(copy.deepcopy(target_action[:6]), initial_pose)[0]
                    if self.monitor:
                        self.times.append(count)
                        for j in range(6):
                            self.state_points[j].append(target_state_monitor[j])
                            self.action_points[j].append(target_action_monitor[j])
                    if self.early_stop:
                        # for j in range(6):
                        self.state_list.append(target_state_monitor)

                if count % 26 == 0:
                    # A_record = time.time()
                    if self.record_trajectory:
                            out_fn = f"test_{next_num}_record.json"
                            out_path = os.path.join(self.record_trajectory, out_fn)
                            with open(out_path, 'w') as f:
                                for record in self.trajectory_data:
                                    f.write(json.dumps(record) + "\n")
                    # print(f"writing json time: {time.time() - A_record}")

                # if count >= 40: # tying
                # if count >= 50:  # we hard code the ending of policy: # insert 50
                if count >= 100:  # we hard code the ending of policy: # insert 50
                    break_policy = True
                    history_states = np.array(self.state_list)
                    for i in range(6):
                        # if np.max(history_states[i, -30:]) - np.min( history_states[i, -30:]) > 0.015:  #the last tying 2 (20)
                        if np.max(history_states[-60:, i]) - np.min(history_states[-60:, i]) > 0.01: #  / insert 5 (50) second the arm is almost static

                            # torch.cuda.synchronize();
                            # policy_end_time = time.time()
                            # print(f"policy runtime: {policy_end_time - policy_start_time}")
                            break_policy = False
                            break

                    if break_policy:
                        # open gripper _test2
                        target_action[6] = -1 # -1 open gripper
                        for i in range(10):
                            self.robot_interface.control(
                                controller_type=self.controller_type,
                                action=target_action,
                                controller_cfg=self.controller_cfg,
                            )
                        print("Rebar is static, Policy finished...")
                        time.sleep(1)
                        break

                if count >= 220:
                    target_action[6] = -1  # -1 open gripper
                    for i in range(10):
                        self.robot_interface.control(
                            controller_type=self.controller_type,
                            action=target_action,
                            controller_cfg=self.controller_cfg,
                        )
                    print("It is over the maximum step limit, Policy finished...")
                    time.sleep(1)
                    break

            # if  self.prim == 1:
            if self.policy_name == "ACT_ZMO_CR":
                self.last_crop_masked_obs = curr_images[0, [0, 3]].cpu().numpy() # BGR tensor

        if self.record_trajectory:
            choice = input("Test success? Enter 1=good, 2=poor, 3=need reinsert, 0=don’t save: ")
            if choice in ('1', '2', "3"):
                if choice == '1':
                    suffix = 'good'
                elif choice == "3":
                    suffix = 'reinsert'
                else:
                    suffix = 'poor'
                out_fn = f"test_{next_num}_{suffix}.json"
                out_path = os.path.join(self.record_trajectory, out_fn)
                with open(out_path, 'w') as f:
                    for record in self.trajectory_data:
                        f.write(json.dumps(record) + "\n")
                print(f"Saved trajectory to {out_path}")
            else:
                print("Trajectory not saved.")

    def close(self):
        self.robot_interface.close()
        del self.robot_interface

    def capture_a_frame(self, binary_detector = False):
        self.obs_processor.get_real_robot_img_obs(crop_option=self.crop_option, crop_list=self.crop_list)
        obs = self.obs_processor.obs
        if binary_detector:

            result_queue = Queue()
            process_start = time.time()
            print("we start to get the last frame...")
            inference_process = Process(target=send_images_to_server,
                                        args=(
                                        6000, result_queue, self.obs_cfg.camera_names, self.rebar_type, obs, process_start,
                                        0, self.slot_name),
                                        daemon=True)
            inference_process.start()
            while result_queue.empty():
                print("we are waiting for the last frame...")
                last_masks_list = result_queue.get()
                curr_images = self.Crop_module.mask_obs(obs, self.obs_cfg.camera_names, last_masks_list, slot_name=self.slot_name)
                curr_images = torch.cat(curr_images, dim=1)
                break

            self.last_crop_masked_obs = curr_images[0, [0, 3]].cpu().numpy() # (C,H,W)
            save_curr_images(10000, curr_images, save_path=os.path.join(self.record_trajectory, str(time.time())))
            print("we get the last frame...")
        else:
            for j in range(2):
                curr_image = obs[f"agentview{j}_rgb"]
                # curr_image = cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(f"act/data/ref_imgs_sam2/rebar1_webcam_{j+1}_scenario.jpg", curr_image)
                if success:
                    print(f"image saved at act/data/ref_imgs_sam2/rebar1_webcam_{j+1}_scenario.jpg")

    def state_detector(self):
        # Failure Recovery
        ResNet_reult_queue = Queue()
        process_start = time.time()
        count = 0
        ResNet_process = Process(target=send_images_to_server,
                                 args=(5000, ResNet_reult_queue, self.obs_cfg.camera_names, self.rebar_type, self.last_crop_masked_obs,
                                       process_start, count, self.slot_name),
                                 daemon=True)
        ResNet_process.start()
        while ResNet_reult_queue.empty():
            success_list = ResNet_reult_queue.get()
            return success_list # {"webcam_1": label, "webcam_2": label}

if __name__ == "__main__":
    import yaml
    yaml_dict = {"ACT_ZMO_CR": "act/ckpt_trained/Bacth_download/configs/insert_v9.yaml",
                 "ACT_ZMO_CR_cropped_horizon1": "act/ckpt_trained/Bacth_download/configs/insert_v9_CR_cropped_horizon_1.yaml",
                 "ACT_ZMO_CR_cropped_horizon2": "act/ckpt_trained/Bacth_download/configs/insert_v9_CR_cropped_horizon_2.yaml",
                 "ACT_ZMO_without_CR": "act/ckpt_trained/Bacth_download/configs/insert_v9_wo_CR.yaml",
                 "ACT_ZMO_CR_qpos_horizon_1": "act/ckpt_trained/Bacth_download/configs/insert_v9_qpos_horizon_1.yaml",

                 "ACT_ZMO_CR_Absolute": "act/ckpt_trained/Bacth_download/configs/insert_v9_absolute.yaml",
                 "ACT_vanilla": "act/ckpt_trained/Bacth_download/configs/insert_v9_vanilla_ACT.yaml",
                 "ACT_vanilla_CR_cropped": "act/ckpt_trained/Bacth_download/configs/insert_v9_vanilla_ACT_CR_cropped.yaml",
                 "ACT_vanilla_CR_cropped_horizon1": "act/ckpt_trained/Bacth_download/configs/insert_v9_vanilla_ACT_CR_cropped_horizon1.yaml",
                 "ACT_vanilla_CR_cropped_horizon2": "act/ckpt_trained/Bacth_download/configs/insert_v9_vanilla_ACT_CR_cropped_horizon2.yaml",

                 "ACT_vanilla_without_Collision_Recovery": "act/ckpt_trained/Bacth_download/configs/insert_v9_vanilla_ACT_wo_CR.yaml",
                 "ACT_ZMO_reinsert": "act/ckpt_trained/Bacth_download/configs/reinsert_v9.yaml", }
    task = "insert" # Reinsert
    # task = "Reinsert"
    # -===================================== our insert policy ===================================== - #
    if task == "insert":
        # TODO: [Important!!!] DO not forget to change the get_points function!
        POLICY = "ACT_ZMO_CR"
        # POLICY = "ACT_ZMO_CR_qpos_horizon_1"
        # POLICY = "ACT_ZMO_CR_cropped_horizon1"
        # POLICY = "ACT_ZMO_CR_cropped_horizon2"

        # POLICY = "ACT_vanilla"
        # POLICY = "ACT_vanilla_CR_cropped_horizon1"
        # POLICY = "ACT_vanilla_CR_cropped_horizon2"

        # POLICY = "ACT_vanilla_without_Collision_Recovery" #   "ACT_ZMO_without_CR" "ACT_ZMO_CR_qpos_horizon_1"
        # POLICY = "ACT_ZMO_without_CR"
        # POLICY = "ACT_ZMO_CR_Absolute"

        with open(yaml_dict[POLICY], "r") as f:
            cfg = yaml.safe_load(f)
        policy_config = cfg["policy_config"]
        policy_args = cfg["policy"]

        # overwrite for convenience
        policy_args["target"] = 0
        policy_args["prim"] = 1
        policy_args["rebar_type"] = "rebar1"
        # policy_args["slot_name"] = "scenario1" # "slot4"# "slot5"
        policy_args["slot_name"] = "slot5"
        # policy_args["crop_option"] = False # capture a image only
        policy = Act_policy(
            **policy_args,
            policy_name = POLICY,
            policy_config=policy_config)

        final_obs = policy.run()
        # policy.capture_a_frame(binary_detector=False)
        policy.close()
    # -===================================== Reinsert ===================================== - #
    else:
        # Choose the policy
        policy = "ACT_ZMO_reinsert" # "ACT_ZMO_CR_qpos_horizon_1"
        with open(yaml_dict[policy], "r") as f:
            cfg = yaml.safe_load(f)
        policy_config = cfg["policy_config"]
        policy_args = cfg["policy"]
        # TODO:  overwrite for convenience
        policy_args["target"] = 0
        policy_args["prim"] = 1
        policy_args["rebar_type"] = "rebar1"
        policy_args["slot_name"] = "slot5"
        FR_policy = Act_policy(
            **policy_args,
            policy_config=policy_config
        )
        FR_policy.capture_a_frame(binary_detector = True) # just for test state_detector
        success_list = FR_policy.state_detector() # we may want to add API for ckp here
        if False in success_list.values():
            print("[Detector:] starting to reinsert...")
            success_list = FR_policy.run()
            FR_policy.close()
        else:
            print("[Detector:] rebar has been fully inserted...")



    # policy_config = {'lr': 1e-5,
    #                  'kl_weight': 10,
    #                  'hidden_dim': 512,
    #                  'dim_feedforward': 3200,
    #                  'lr_backbone': 1e-5,
    #                  'backbone': 'resnet18',
    #                  'enc_layers': 4,
    #                  'dec_layers': 1,
    #                  'nheads': 8,
    #
    #                  'num_queries': 10,
    #                  'camera_names': ["webcam_1_crop1", "webcam_1_crop2", "webcam_1_crop3", "webcam_2_crop1",
    #                                   "webcam_2_crop2", "webcam_2_crop3"],
    #                  # 'camera_names': ["webcam_1", "webcam_2"],
    #                  "input_type": ["images", "target", "primitive", "qpos"],
    #                  "relative_bool": True,
    #                  "prim_range": ["1_insert"],  # TODO: if list includes more than 1 element, "self.stats = self.stats[self.policy_config["prim_range"]]" this should be modified ["0_grasp"] # ["2_Reinsert"] # ["3_tying"]
    #                  'img_horizon': 1,
    #                  'qpos_horizon': 5}

    # policy = Act_policy(target = 0, prim = 1, # 1 means ["1_insert"]
    #                     checkpoint_dir= "act/ckpt_trained/insert_v9/policy_last.ckpt", # BACKUP/
    #                     normalize_dir = "act/ckpt_trained/insert_v9/dataset_stats.pkl", # BACKUP/
    #                     rebar_type="rebar1",
    #                     slot_name = "slot5", # slot5
    #                     crop_option=True,
    #                     policy_config = policy_config,
    #                     monitor_option = True,
    #                     sam_option=True,
    #                     record_trajectory= "/mnt/sda1/Dropbox/Dropbox/paper3_insert_rollout_data/insert_v9",
    #                     )

    # -===================================== Vanilla ACT ===================================== - #
    # policy_config = {'lr': 1e-5,
    #                  'kl_weight': 10,
    #                  'hidden_dim': 512,
    #                  'dim_feedforward': 3200,
    #                  'lr_backbone': 1e-5,
    #                  'backbone': 'resnet18',
    #                  'enc_layers': 4,
    #                  'dec_layers': 1,
    #                  'nheads': 8,
    #
    #                  'num_queries': 10,
    #                  'camera_names': ["webcam_1", "webcam_2"],
    #                  "input_type": ["images", "target", "primitive", "qpos"],
    #                  "relative_bool": True,
    #                  "prim_range": ["1_insert"],  # TODO: if list includes more than 1 element, "self.stats = self.stats[self.policy_config["prim_range"]]" this should be modified ["0_grasp"] # ["2_Reinsert"] # ["3_tying"]
    #                  'img_horizon': 1,
    #                  'qpos_horizon': 5}
    #
    # policy = Act_policy(target = 0, prim = 1, # 1 means ["1_insert"]/
    #                     checkpoint_dir= "act/ckpt_trained/batch_download/insert_v9_vanilla_ACT/policy_epoch_2000_seed_0.ckpt", # BACKUP/
    #                     normalize_dir = "act/ckpt_trained/batch_download/insert_v9_vanilla_ACT/dataset_stats.pkl", # BACKUP/
    #                     rebar_type="rebar1",
    #                     slot_name = "slot5", # slot5/
    #                     crop_option=True,
    #                     policy_config = policy_config,
    #                     monitor_option = True,
    #                     sam_option=False,
    #                     record_trajectory= "/mnt/sda1/Dropbox/Dropbox/paper3_insert_rollout_data/insert_v9_act_vanilla",
    #                     )

    # final_obs = policy.run()
    # policy.close()



    # policy_config = {'lr': 1e-5,
    #                  'kl_weight': 10,
    #                  'hidden_dim': 512,
    #                  'dim_feedforward': 3200,
    #                  'lr_backbone': 1e-5,
    #                  'backbone': 'resnet18',
    #                  'enc_layers': 4,
    #                  'dec_layers': 1,
    #                  'nheads': 8,
    #
    #                  'num_queries': 10,
    #                  'camera_names': ["webcam_1", "webcam_2"],
    #                  "input_type": ["images", "target", "primitive", "qpos"],
    #                  "relative_bool": True,
    #                  "prim_range": ["3_tying"],  # TODO: if list includes more than 1 element, "self.stats = self.stats[self.policy_config["prim_range"]]" this should be modified ["0_grasp"] # ["2_Reinsert"] # ["3_tying"]
    #                  'img_horizon': 1,
    #                  'qpos_horizon': 5}
    #
    # policy = Act_policy(target = 0, prim = 3, # 3 means ["3_tying"]
    #                     checkpoint_dir= "act/ckpt_trained/tying_v3_final/policy_last.ckpt",
    #                     normalize_dir = "act/ckpt_trained/tying_v3_final/dataset_stats.pkl",
    #                     # checkpoint_dir="act/ckpt_trained/tying_v3/policy_last.ckpt",
    #                     # normalize_dir="act/ckpt_trained/tying_v3/dataset_stats.pkl",
    #                     rebar_type="rebar1",
    #                     slot_name = "tying",
    #                     crop_option=True, # True False
    #                     policy_config = policy_config,
    #                     monitor_option = False,
    #                     sam_option=True,
    #                     record_trajectory= "/mnt/sda1/Dropbox/Dropbox/paper3_tying_rollout_data",
    #                     )
    # # policy.capture_a_frame()  # just for test state_detector
    # final_obs = policy.run()
    # policy.close()

    # FR_config = policy_config.copy()
    # FR_config["prim_range"] = ["1_insert"]
    # FR_policy = Act_policy(checkpoint_dir="act/ckpt_trained/reinsert_v9/policy_last.ckpt",
    #                        normalize_dir="act/ckpt_trained/reinsert_v9/dataset_stats.pkl",
    #                        policy_config=FR_config)
    # FR_policy.capture_a_frame() # just for test state_detector
    # success_list = FR_policy.state_detector() # we may want to add API for ckp here
    # if False in success_list.values():
    #     # adjust insertion
    #     success_list = FR_policy.run()
    #     FR_policy.close()
    # else:
    #     print("Finish...")