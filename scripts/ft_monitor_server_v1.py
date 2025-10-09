from multiprocessing.connection import Listener
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

    def send_data(self, data):
        # Ensure we're sending CPU-native serializable data (avoid CUDA IPC)
        if isinstance(data, torch.Tensor):
            try:
                payload = data.detach().to('cpu').reshape(-1).tolist()
            except Exception:
                payload = data.detach().to('cpu').numpy().reshape(-1).tolist()
        else:
            payload = np.asarray(data).reshape(-1).tolist()
        with Client(self.address, authkey=self.authkey) as conn:
            message = {"ft_data": payload}
            conn.send(message)
            result = conn.recv()
            # Removed undefined result_queue usage
            return result

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
        print(f"FT data: {data}")
        # Ensure 1D numpy array of length 6
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)
        arr = arr.reshape(-1)
        if arr.size != 6:
            return data
        # Update buffers
        t = len(self.timesteps)
        self.timesteps.append(t)
        for i in range(6):
            self.values[i].append(float(arr[i]))
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
                        result = self.plot_ft(message["ft_data"])
                        torch.cuda.synchronize()
                        # st =  message["process_start"]
                        # print(f"PROCESSS using FoundationStereo: {time.time() - st}")
                        conn.send(result)
                except Exception as e:
                    print("Connection closed:", e)
                finally:
                    conn.close()

if __name__ == '__main__':
    server = PlotFTServer(address=('localhost', 6002), authkey=b'secret')
    server.run()