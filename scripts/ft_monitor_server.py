from multiprocessing.connection import Listener
from tarfile import data_filter
import torch
from docutils.nodes import address
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.connection import Client

class PlotFTClient():
    def __init__(self, address = ('localhost', 6000), authkey = b"secret"):
        self.address = address
        self.authkey = authkey
        import ipdb; ipdb.set_trace()
        self.conn = Client(self.address, authkey=self.authkey)  # persistent

    def send_data(self, message):
        self.conn.send(message)

class PlotFTServer():
    def __init__(self,address, authkey):
        self.address = address
        self.authkey = authkey
        self.init_plot()
    
    def init_plot(self):
        # Initialize time series buffers
        self.timesteps = []
        self.values = [[] for _ in range(6)]  # Fx, Fy, Fz, Tx, Ty, Tz
        self.labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        # Setup matplotlib for live updating
        plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        self.axes = axes.ravel()
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], lw=1.5)
            ax.set_ylabel(self.labels[i])
            ax.grid(True, linestyle='--', alpha=0.4)
            self.lines.append(line)
        self.axes[0].set_title("Force/Torque over time")
        for ax in self.axes[3:]:
            ax.set_xlabel("timestep")
        self.fig.tight_layout()

    def plot_ft(self, data):
        data = np.asarray(data)
        print(f"FT data: {data}")
        # print(f"FT data: {data}")
        if data.size != 6:
            return data
        # Update buffers
        t = len(self.timesteps)
        self.timesteps.append(t)
        for i in range(6):
            self.values[i].append(float(data[i]))
        # Refresh plot lines
        for i in range(6):
            self.lines[i].set_data(self.timesteps, self.values[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return data

    def run(self):
        with Listener(self.address, authkey=self.authkey) as listener:
            print("Inference server is running at", self.address)
            while True:
                conn = listener.accept()
                print("Connected to client:", listener.last_accepted)
                try:
                    while True:
                        message = conn.recv()  # Receive image data
                        print("Server received!")
                        print(f"Delivery running time: {time.time() - message['time']}")
                        # result = self.plot_ft(message["ft_data"]) 
                        # conn.send(result)
                except Exception as e:
                    print("Connection closed:", e)
                finally:
                    conn.close()

if __name__ == '__main__':
    server = PlotFTServer(address=('localhost', 6002), authkey=b'secret')
    server.run()