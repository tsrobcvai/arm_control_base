# utils/tty_keys.py
import sys, termios, tty, select, atexit

class KeyWatcher:
    def __enter__(self):
        if not sys.stdin.isatty():
            # Not attached to a TTY (e.g., service/nohup). Can't read keys this way.
            self.enabled = False
            return self
        self.enabled = True
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # single-keystroke mode
        atexit.register(self.restore)
        return self

    def __exit__(self, *args):
        self.restore()

    def restore(self):
        try:
            if getattr(self, "enabled", False):
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass

    def poll(self):  # returns a single char or None
        if not self.enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if r else None
