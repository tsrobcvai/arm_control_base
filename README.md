# UFACTORY TELEOPERATION

**Experiments for Ufactory Xarm Robots**

## 1. Installation
```bash
conda create -n ufact python=3.9
conda activate ufact
# install ufactory API, Install from source code
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
pip install .
pip install -r requirements.txt
cd ..
cd io_devices/oculus_reader
pip install -r requirements.txt
cd ../..
```

### 2. Meta quest 2
### Installation

Headset should be plugged into PC, and the permissions prompt "USB debugging" should be accepted in the system of headset. Also, the controller has to be in view of the headset cameras. It detects the pose of the handle via infrared stickers on the handle. Teleoperation works by applying the changes to the oculus handle’s pose to the robot gripper’s pose.

### Core code

This repo supports Meta Quest 2 as the teleoperation device. To start, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader/blob/main/oculus_reader/reader.py) by following instructions in the link.

["meta_quest2.py"](io_devices/meta_quest2.py)


### Operation instruction:
* Move: **Move controller**
* Grasp: **Right trigger**
* Over: **A**

## 3. Config setting

...

## 4. Collecting demos


Example usage (note that `camera_node.py` and `data_collection_metaquest.py` should be run simultaneously in at least two separate windows):


### 4.1 Reset robot pose if need


For Ufactory XArm6 (default):

```bash
 # Reset robot joints to zero position
 python scripts/reset_robot_joints.py 
 # Reset robot Cartesian pose to zero position
 python scripts/reset_robot_cartesian.py
```

For other robots (i.e. Ufactory lite6)

```bash
# Reset robot joints to zero position
python scripts/reset_robot_joints.py  --robot lite6 --ip 192.168.1.193
# Reset robot Cartesian pose to zero position
python scripts/reset_robot_cartesian.py  --robot lite6 --ip 192.168.1.193
```

### 4.2 Open cameras needed, such as

```bash
# check cameras path
v4l2-ctl --list-devices


# Launch Webcam camera node
python scripts/camera_node.py --camera-ref webcam_0 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --camera-address '/dev/video0'

# Launch RealSense camera node
python scripts/camera_node.py --camera-ref rs_1 --use-rgb  --visualization --img-h 480 --img-w 640 --fps 30
```
** if you want to use RGB and depth data from RealSense camera, you can run the following command:**

```bash
# RealSense camera example with RGB and depth
python scripts/camera_node.py --camera-ref rs_1 --use-rgb --use-depth --visualization --img-h 480 --img-w 640 --fps 30
```

Node processes can be cleaned up by running `pkill -9 -f scripts/camera_node.py`.

### 4.3 Start collecting

For Ufactory XArm6 (default):

```bash
# Start collecting data with Meta Quest 2
python scripts/data_collection_metaquest.py
```

For other robots (i.e. Ufactory lite6)

```bash
# Start collecting data with Meta Quest 2
python scripts/data_collection_metaquest.py --robot lite6 --ip 192.168.1.193
```

## 5. Data format

Collected data is stored in the `demos_collected` folder, with each run in a separate subfolder named `runXXX`, where `XXX` is the run number. Each run folder contains:

### 5.1 Folder Structure

```
demos_collected/
├── run001/
│   ├── config.json                    # Configuration file
│   ├── demo_action.npz               # Action sequence data
│   ├── demo_ee_states.npz            # End effector states
│   ├── demo_target_pose_mat.npz      # Target pose matrix
│   ├── demo_joint_states.npz         # Joint states
│   ├── demo_gripper_states.npz       # Gripper states
│   ├── demo_action_hot.npz           # Action hotkey states
│   └── demo_camera_1.npz             # Camera data index
├── run002/
│   ├── config.json
│   ├── demo_*.npz
│   └── demo_camera_1.npz
└── images/
    └── rs_rs_1_<timestamp>/
        ├── color_000000001.jpg        # run001: i.e. Image 1-50
        ├── color_000000002.jpg
        ├── ...
        ├── color_000000051.jpg        # run002: i.e. Image 51-120
        └── ...
```                                                                                                                                                                                                               

# Acknowledgement
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot

### debug