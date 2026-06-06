import time
from threading import Event, Lock, Thread


class LiveFrameReader:
    def __init__(self, capture):
        self.capture = capture
        self.lock = Lock()
        self.stop_event = Event()
        self.frame = None
        self.version = 0
        self.timestamp = 0.0
        self.last_ok_at = time.monotonic()
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def read(self, last_version=0, timeout_seconds=0.25):
        deadline = time.monotonic() + timeout_seconds
        while not self.stop_event.is_set():
            with self.lock:
                if self.frame is not None and self.version != last_version:
                    return True, self.frame.copy(), self.timestamp, self.version

            if time.monotonic() >= deadline:
                with self.lock:
                    if self.frame is not None:
                        return True, self.frame.copy(), self.timestamp, self.version
                return False, None, 0.0, last_version

            time.sleep(0.003)

        return False, None, 0.0, last_version

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=0.5)
        self.capture.release()

    def _read_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.02)
                continue

            timestamp = time.monotonic()
            with self.lock:
                self.frame = frame
                self.version += 1
                self.timestamp = timestamp
                self.last_ok_at = timestamp
