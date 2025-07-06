import time
import numpy as np
from xarm.wrapper import XArmAPI


def main(robot: str = "xarm6",
         ip: str = "192.168.1.235",
         pose: list[float] = None):

    pose = np.array(pose, dtype=np.float32)

    print(f"Connecting to {robot} at {ip} ...")
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.clean_error()

    # manual mode first

    # arm.set_mode(2)
    # arm.set_state(0)
    
    # # ask operator for confirmation
    # resp = input("Reset the arm to this pose? (y to confirm): ").strip().lower()
    # if resp != "y":
    #     print("Aborted by user.")
    #     arm.disconnect()
    #     return

    # switch to position control (mode 0) and move
    print(f"current position: {np.asarray(arm.get_position_aa(is_radian=True)[1])}")
    import pdb; pdb.set_trace()
    
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"Moving to pose: {pose} (units: mm, deg)")
    arm.set_position(*pose, wait=True) 

    arm.disconnect()
    print("Reset finished.")

# ==========================================================================
# Configure your robot here
# We will get in to the manual mode first, after the user confirms, we will switch to position control mode
# ==========================================================================
if __name__ == "__main__":
    ROBOT = "xarm6"                 # "xarm6" | "lite6" | "uf850"
    IP    = "192.168.1.235"                    # None → use default for ROBOT
    POSE  = [481, 48.4, 350, -180, -2, 3.6]                   # None → use robot’s default pose

    # POSE example: [300, 50, 350, -180, 0, 0]

    main(robot=ROBOT, ip=IP, pose=POSE)
