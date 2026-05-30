import math
import time


class SpeedEstimator:
    def __init__(self, pixels_per_meter=45.0, smoothing=0.35, max_history_age=3.0):
        self.pixels_per_meter = max(1.0, pixels_per_meter)
        self.smoothing = min(max(smoothing, 0.0), 1.0)
        self.max_history_age = max_history_age
        self.history = {}

    def update(self, detection, timestamp=None):
        track_id = detection.get("track_id")
        if track_id is None:
            detection["speed_kmh"] = None
            return detection

        timestamp = timestamp if timestamp is not None else time.monotonic()
        center = self._center(detection)
        previous = self.history.get(track_id)
        speed_kmh = 0.0

        if previous:
            dt = max(0.001, timestamp - previous["timestamp"])
            distance_px = math.dist(center, previous["center"])
            instant_kmh = (distance_px / self.pixels_per_meter) / dt * 3.6
            speed_kmh = (
                previous["speed_kmh"] * (1 - self.smoothing)
                + instant_kmh * self.smoothing
            )

        self.history[track_id] = {
            "center": center,
            "timestamp": timestamp,
            "speed_kmh": speed_kmh,
            "seen_at": time.monotonic(),
        }
        detection["speed_kmh"] = speed_kmh
        self._prune()
        return detection

    def reset(self):
        self.history.clear()

    @staticmethod
    def _center(detection):
        return (
            (detection["x1"] + detection["x2"]) / 2,
            (detection["y1"] + detection["y2"]) / 2,
        )

    def _prune(self):
        now = time.monotonic()
        expired = [
            track_id
            for track_id, item in self.history.items()
            if now - item["seen_at"] > self.max_history_age
        ]
        for track_id in expired:
            self.history.pop(track_id, None)
