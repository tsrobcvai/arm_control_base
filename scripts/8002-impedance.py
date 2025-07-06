#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: impedance control with the end force sensor
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

ip = "192.168.1.235"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.clean_error()
arm.set_mode(0)
arm.set_state(0)
time.sleep(0.1)

# set tool impedance parameters:
K_pos = 0.01         #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
K_ori = 0.01          #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad)

# Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful. 
M = float(0.05)  # 0.06  x/y/z equivalent mass; range: 0.02 ~ 1 kg
J = M * 0.05     #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

c_axis = [1,1,1,0,0,0] # set z axis as compliant axis
ref_frame = 0         # 0 : base , 1 : tool

import math
def crit_b(m, k, zeta=50):
    return 200000 * zeta * math.sqrt(m * k)

B_xyz = [crit_b(M, K_pos)]*3
B_rpy = [crit_b(J, K_ori)]*3
B = B_xyz + B_rpy   

arm.set_impedance_mbk([M, M, M, J, J, J], [K_pos, K_pos, K_pos, K_ori, K_ori, K_ori], B ) # B [0]*6
arm.set_impedance_config(ref_frame, c_axis)

# enable ft sensor communication
arm.ft_sensor_enable(1)
# will overwrite previous sensor zero and payload configuration
arm.ft_sensor_set_zero() # remove this if zero_offset and payload already identified & compensated!
time.sleep(0.2) # wait for writing zero operation to take effect, do not remove

# move robot in impedance control application
arm.ft_sensor_app_set(1)
# will start after set_state(0)
arm.set_state(0)

# compliance effective for 10 secs, you may send position command to robot, or just keep it still
time.sleep(40)

# remember to reset ft_sensor_app when finished
arm.ft_sensor_app_set(0)
arm.ft_sensor_enable(0)
arm.disconnect()
