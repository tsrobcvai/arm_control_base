'''
Base on TCP port 30000, 250HZ.
Firmware version should be 2.1.101 or later.
Here is an example to get actual joint currents by TCP 30000 port, unit: Ampere.
'''

import socket
import struct
from arm_control_base.scripts.ft_monitor_server_v1 import PlotFTClient

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
        raw_ft_data = bytes_to_fp32_list(data[689:712]) # N & N·m[Fx,Fy,Fz,Tx,Ty,Tz]
        processed_ft_data = bytes_to_fp32_list(data[713:736]) # N & N·m[Fx,Fy,Fz,Tx,Ty,Tz]
        return raw_ft_data, processed_ft_data

if __name__ == "__main__":
    # A example to read and plot FT sensor readings
    import time
    ft_plot_client = PlotFTClient(address=('localhost', 6002), authkey=b'secret')
    ft_reader = robot_port_reader(robot_ip = '192.168.1.235', robot_port = 30000)
    cnt=0
    start_time = time.time()
    while cnt < 200:
        st = start_time if cnt == 0 else time.time()
        raw_ft_data, processed_ft_data = ft_reader.read_ft_data()
        # ft_plot_client.send_data({"ft_data": processed_ft_data})
        print("processed_ft_data={}".format(processed_ft_data))
        cnt+=1
        # if time.time() - st < 0.02:
        #     time.sleep(0.02 - (time.time() - st))
    end_time = time.time()
    print("time cost={}".format(end_time - start_time))
    print("fps={}".format(cnt/(end_time - start_time)))