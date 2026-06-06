import time
from threading import Lock


class LightweightTracker:
    _PRUNE_AGE = 4.0
    _MIN_IOU = 0.02
    _DISTANCE_BASE = 180.0
    _IOU_WEIGHT = 1.4
    _DIST_WEIGHT = 0.65

    def __init__(self):
        self._tracks = {}
        self._next_id = 1
        self._lock = Lock()

    def assign(self, detections, timestamp):
        with self._lock:
            used = set()
            for detection in sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True):
                track_id = self._match(detection, used)
                if track_id is None:
                    track_id = self._next_id
                    self._next_id += 1
                detection["track_id"] = track_id
                used.add(track_id)
                self._tracks[track_id] = {
                    "box": _box_tuple(detection),
                    "center": _center(detection),
                    "timestamp": timestamp,
                    "seen_at": time.monotonic(),
                }
            self._prune()
        return detections

    def count(self):
        with self._lock:
            return len(self._tracks)

    def clear(self):
        with self._lock:
            self._tracks.clear()
            self._next_id = 1

    def _match(self, detection, used):
        best_id = None
        best_score = 0.0
        box = _box_tuple(detection)
        center = _center(detection)
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        distance_limit = max(self._DISTANCE_BASE, (width + height) * 2.0)

        for track_id, track in self._tracks.items():
            if track_id in used:
                continue
            iou = _box_iou(box, track["box"])
            distance = (
                (center[0] - track["center"][0]) ** 2
                + (center[1] - track["center"][1]) ** 2
            ) ** 0.5
            if iou < self._MIN_IOU and distance > distance_limit:
                continue
            score = iou * self._IOU_WEIGHT + max(0.0, 1.0 - distance / distance_limit) * self._DIST_WEIGHT
            if score > best_score:
                best_score = score
                best_id = track_id

        return best_id

    def _prune(self):
        now = time.monotonic()
        expired = [tid for tid, t in self._tracks.items() if now - t["seen_at"] > self._PRUNE_AGE]
        for tid in expired:
            self._tracks.pop(tid, None)


def _box_tuple(detection):
    return (
        float(detection["x1"]),
        float(detection["y1"]),
        float(detection["x2"]),
        float(detection["y2"]),
    )


def _center(detection):
    return (
        (float(detection["x1"]) + float(detection["x2"])) / 2,
        (float(detection["y1"]) + float(detection["y2"])) / 2,
    )


def _box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    first_area = max(1.0, first[2] - first[0]) * max(1.0, first[3] - first[1])
    second_area = max(1.0, second[2] - second[0]) * max(1.0, second[3] - second[1])
    return inter / (first_area + second_area - inter)
