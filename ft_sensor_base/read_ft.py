'''
Base on TCP port 30000, 250HZ.
Firmware version should be 2.1.101 or later.
Here is an example to get actual joint currents by TCP 30000 port, unit: Ampere.
'''

import socket
import struct

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
    def __init__(self, robot_ip = '192.168.1.77'):
        self.robot_ip = robot_ip
        self.robot_port = 30000
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
    robot_ip = '192.168.1.77'
    robot_port = 30000
    # create socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)
    sock.settimeout(1)
    sock.connect((robot_ip, robot_port))

    buffer = sock.recv(4)
    print(buffer)
    while len(buffer) < 4:
        buffer += sock.recv(4 - len(buffer))
    size = bytes_to_u32(buffer[:4])

    cnt=0
    while cnt<1000:
        buffer += sock.recv(size - len(buffer))
        if len(buffer) < size:
            continue
        cnt+=1
        data = buffer[:size]
        buffer = buffer[size:]
        # actual_joint_currents=bytes_to_fp32_list(data[200:228])
        processed_ft_data = bytes_to_fp32_list(data[713:736]) # N & N·m[Fx,Fy,Fz,Tx,Ty,Tz]
        print("counter={},processed_ft_data={}".format(cnt,processed_ft_data))