'''
Base on TCP port 30000, 250HZ.
Firmware version should be 2.1.101 or later.
Here is an example to get actual joint currents by TCP 30000 port, unit: Ampere.
'''

import socket
import struct
from arm_control_base.scripts.ft_monitor_server import PlotFTClient
from xarm.wrapper import XArmAPI
import time
import numpy as np
from multiprocessing.connection import Listener
import threading

def bytes_to_fp32(bytes_data, is_big_endian=False):
    return struct.unpack('>f' if is_big_endian else '<f', bytes_data)[0]

def bytes_to_fp32_list(bytes_data, n=0, is_big_endian=False):
    ret = []
    count = n if n > 0 else len(bytes_data) // 4
    for i in range(count):
        ret.append(bytes_to_fp32(bytes_data[i * 4: i * 4 + 4], is_big_endian))
    return ret

def bytes_to_u32(data):
    data_u32 = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3]
    return data_u32

class robot_port_reader():
    def __init__(self, robot_ip = '192.168.1.235', robot_port = 30000):
        # init arm
        self.arm = XArmAPI(robot_ip, is_radian=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

        # Initialize the FT sensor
        self.arm.ft_sensor_enable(1)
        self.arm.ft_sensor_set_zero()
        time.sleep(0.2)
        self.arm.ft_sensor_app_set(1)
        self.arm.set_state(0)

        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(True)
        self.sock.settimeout(1)
        self.sock.connect((self.robot_ip, self.robot_port))

    def read_ft_data(self):
        buffer = self.sock.recv(4)
        while len(buffer) < 4:
            buffer += self.sock.recv(4 - len(buffer))
        size = bytes_to_u32(buffer[:4])

        buffer = self.sock.recv(size)
        data = buffer[:size]

        buffer = buffer[size:]
        raw_ft_data = bytes_to_fp32_list(data[688:712]) # N & N·m[Fx,Fy,Fz,Tx,Ty,Tz]
        # import pdb; pdb.set_trace()
        processed_ft_data = bytes_to_fp32_list(data[712:736]) # N & N·m[Fx,Fy,Fz,Tx,Ty,Tz]
        return raw_ft_data, processed_ft_data

class FTServer():
    def __init__(self,address, authkey):
        self.address = address
        self.authkey = authkey
        self.ft_reader = robot_port_reader(robot_ip = '192.168.1.235', robot_port = 30000)
        self.ft_data = None

        self.run_ft_reader_thread = threading.Thread(target=self.run_ft_reader)
        self.run_ft_reader_thread.start()

    def run_ft_reader(self):
        while True:
            raw_ft_data, processed_ft_data = self.ft_reader.read_ft_data()
            self.ft_data = processed_ft_data

    def run(self):
        with Listener(self.address, authkey=self.authkey) as listener:
            print("Inference server is running at", self.address)
            while True:
                conn = listener.accept()
                print("Connected to client:", listener.last_accepted)
                try:
                    while True:
                        message = conn.recv()  # Receive image data 
                        if message is not None:
                            conn.send(self.ft_data)
                            print(f"FT data: {self.ft_data}")
                except Exception as e:
                    print("Connection closed:", e)
                finally:
                    conn.close()

if __name__ == "__main__":

    # arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_mode", action="store_true", help="run in server mode")  
    args = parser.parse_args()

    if args.server_mode:
        ft_server = FTServer(address=('localhost', 6002), authkey=b'secret')
        ft_server.run()

         
    else:
        # A example to read and plot FT sensor readings
        DECIMALS = 3
        ft_plot_client = PlotFTClient(address=('localhost', 6002), authkey=b'secret')
        ft_reader = robot_port_reader(robot_ip = '192.168.1.235', robot_port = 30000)
        cnt=0
        start_time = time.time()
        # FT_data = []
        while True:
        # while cnt < 200:
            st = start_time if cnt == 0 else time.time()
            raw_ft_data, processed_ft_data = ft_reader.read_ft_data()
            print("processed_ft_data={}".format(
                np.array2string(
                    np.asarray(processed_ft_data),
                    formatter={'float_kind': lambda x: f"{x:.{DECIMALS}f}"}
                )
            ))
            if cnt % 5 == 0:
                ft_plot_client.send_data({"ft_data": processed_ft_data, "cnt": cnt, "time": time.time()})
            # time.sleep(0.02)
            # FT_data.append(processed_ft_data)
            cnt+=1
