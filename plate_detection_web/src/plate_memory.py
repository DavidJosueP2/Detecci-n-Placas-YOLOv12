import time


class PlateMemory:
    TTL = 5.0

    def __init__(self):
        self._data = {}

    @staticmethod
    def key_for(detection):
        track_id = detection.get("track_id")
        if track_id is not None:
            return f"track:{track_id}"
        center_x = (detection["x1"] + detection["x2"]) / 2
        center_y = (detection["y1"] + detection["y2"]) / 2
        return f"pos:{round(center_x / 80)}:{round(center_y / 80)}"

    def get(self, detection):
        cached = self._data.get(self.key_for(detection))
        if cached is None:
            return None
        if time.monotonic() - cached["seen_at"] > self.TTL:
            return None
        cached["seen_at"] = time.monotonic()
        return cached

    def get_raw(self, key):
        return self._data.get(key)

    def store(self, key, entry):
        self._data[key] = entry

    def clear(self):
        self._data.clear()
