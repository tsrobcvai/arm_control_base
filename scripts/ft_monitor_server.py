import os
import re
import time
from multiprocessing.connection import Listener
from typing import List, Optional

import matplotlib
# Try to use an interactive backend; fall back to Agg if display is not available
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FTMonitorServer:
    def __init__(self, address: tuple, authkey: bytes):
        self.address = address
        self.authkey = authkey

        # Buffers
        self.timestamps: List[float] = []
        self.values: List[List[float]] = [[], [], [], [], [], []]  # Fx, Fy, Fz, Tx, Ty, Tz
        self.start_time: Optional[float] = None

        # Matplotlib figure setup
        self.fig = None
        self.axes = None
        self.lines = None
        self.max_window_seconds = 120.0  # keep last 2 minutes for display
        self.channel_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    def _ensure_figure(self):
        if self.fig is not None:
            return
        self.fig, self.axes = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
        self.lines = []
        for idx, ax in enumerate(self.axes.flat):
            line, = ax.plot([], [], lw=1.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(self.channel_names[idx])
            self.lines.append(line)
        plt.ion()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_buffers(self, t_abs: float, ft_values: List[float]):
        if self.start_time is None:
            self.start_time = t_abs
        t_rel = t_abs - self.start_time
        self.timestamps.append(t_rel)
        for i in range(6):
            self.values[i].append(float(ft_values[i]))

        # Trim buffers to max window
        while self.timestamps and (self.timestamps[-1] - self.timestamps[0]) > self.max_window_seconds:
            self.timestamps.pop(0)
            for i in range(6):
                self.values[i].pop(0)

    def _redraw(self):
        if self.fig is None:
            return
        for i, ax in enumerate(self.axes.flat):
            self.lines[i].set_data(self.timestamps, self.values[i])
            # Update axes limits
            if self.timestamps:
                ax.set_xlim(self.timestamps[0], max(self.timestamps[-1], self.timestamps[0] + 1.0))
                y_vals = self.values[i]
                y_min = min(y_vals)
                y_max = max(y_vals)
                if y_min == y_max:
                    pad = 1.0 if y_max == 0.0 else abs(y_max) * 0.1
                    ax.set_ylim(y_min - pad, y_max + pad)
                else:
                    pad = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - pad, y_max + pad)
                # Title with max absolute value
                max_abs = max((abs(v) for v in y_vals), default=0.0)
                ax.set_title(f"{self.channel_names[i]}  max |val| = {max_abs:.2f}")
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            # Headless mode; ignore display errors
            pass

    def _next_id(self, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        regex = re.compile(r"FT_(\d{3})\.jpg$")
        max_id = -1
        try:
            for fname in os.listdir(save_dir):
                m = regex.match(fname)
                if m:
                    idx = int(m.group(1))
                    if idx > max_id:
                        max_id = idx
        except FileNotFoundError:
            pass
        return f"{max_id + 1:03d}"

    def _save_current_figure(self, save_dir: str):
        if not save_dir:
            return
        img_id = self._next_id(save_dir)
        out_path = os.path.join(save_dir, f"FT_{img_id}.jpg")
        try:
            # Keep file reasonably small
            self.fig.savefig(
                out_path,
                dpi=80,
                bbox_inches="tight",
                pad_inches=0.05,
                facecolor="white",
                pil_kwargs={"quality": 70, "optimize": True},
            )
            print(f"Saved FT plot: {out_path}")
        except Exception as e:
            print(f"Failed to save FT plot to {out_path}: {e}")

    def run(self):
        with Listener(self.address, authkey=self.authkey) as listener:
            print("FT monitor server is running at", self.address)
            while True:
                conn = listener.accept()
                print("Connected to client:", listener.last_accepted)
                try:
                    self._ensure_figure()
                    while True:
                        message = conn.recv()
                        # Expected message format:
                        # {"type": "ft", "count": int, "time": float, "ft": [6 floats],
                        #  "save": bool, "save_path": str or None}
                        if not isinstance(message, dict) or message.get("type") != "ft":
                            continue
                        ft_vals = message.get("ft", None)
                        t_now = float(message.get("time", time.time()))
                        if ft_vals is None or len(ft_vals) != 6:
                            continue
                        self._update_buffers(t_now, ft_vals)
                        self._redraw()
                        if message.get("save") and message.get("save_path"):
                            self._save_current_figure(message.get("save_path"))
                        # We intentionally do not send any response to keep the pipeline lightweight
                except EOFError:
                    print("Client disconnected.")
                except Exception as e:
                    print("Connection closed due to error:", e)
                finally:
                    conn.close()


if __name__ == "__main__":
    server = FTMonitorServer(("localhost", 5000), authkey=b"secret")
    server.run()