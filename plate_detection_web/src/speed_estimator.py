import time


class SpeedEstimator:
    def __init__(
        self,
        line_a_y=0.45,
        line_b_y=0.70,
        line_a_right_y=None,
        line_b_right_y=None,
        roi_x1=0.0,
        roi_x2=1.0,
        distance_meters=5.0,
        direction="both",
        hysteresis_px=8.0,
        min_travel_time=0.15,
        max_travel_time=8.0,
        min_movement_px=35.0,
        min_partial_progress=0.35,
        max_history_age=6.0,
    ):
        self.line_a_left_y = self._clamp_normalized(line_a_y)
        self.line_a_right_y = self._clamp_normalized(
            line_a_y if line_a_right_y is None else line_a_right_y
        )
        self.line_b_left_y = self._clamp_normalized(line_b_y)
        self.line_b_right_y = self._clamp_normalized(
            line_b_y if line_b_right_y is None else line_b_right_y
        )
        self.roi_x1 = self._clamp_normalized(roi_x1)
        self.roi_x2 = self._clamp_normalized(roi_x2)
        if self.roi_x2 < self.roi_x1:
            self.roi_x1, self.roi_x2 = self.roi_x2, self.roi_x1
        self.distance_meters = max(0.1, float(distance_meters))
        self.direction = str(direction).strip().lower()
        self.hysteresis_px = max(1.0, float(hysteresis_px))
        self.min_travel_time = max(0.01, float(min_travel_time))
        self.max_travel_time = max(self.min_travel_time, float(max_travel_time))
        self.min_movement_px = max(0.0, float(min_movement_px))
        self.min_partial_progress = max(0.05, min(1.0, float(min_partial_progress)))
        self.max_history_age = max_history_age
        self.history = {}

    def update(self, detection, timestamp=None, frame_shape=None):
        track_id = detection.get("speed_key") or detection.get("track_id")
        if track_id is None or frame_shape is None:
            detection["speed_kmh"] = None
            detection["speed_status"] = "sin seguimiento"
            return detection

        timestamp = timestamp if timestamp is not None else time.monotonic()
        anchor = self._anchor(detection)
        lines = self._line_pixels(frame_shape)
        progress = self._progress_between_lines(anchor, lines)
        if not self._inside_roi(anchor, frame_shape):
            detection["speed_kmh"] = None
            detection["speed_status"] = "fuera de zona"
            self._prune()
            return detection

        state = self.history.get(track_id)

        if state is None:
            self.history[track_id] = self._new_state(anchor, timestamp, lines, progress)
            detection["speed_kmh"] = None
            detection["speed_status"] = self._waiting_status(progress)
            self._prune()
            return detection

        self._update_motion(state, anchor, timestamp)
        self._update_progress(state, progress, timestamp)
        for line_name, line in self._measurement_lines(lines).items():
            self._update_line_crossing(state, line_name, line, anchor, timestamp)
        state["previous_anchor"] = anchor
        state["previous_timestamp"] = timestamp

        speed_kmh = self._calculate_speed_if_ready(state)
        if speed_kmh is None:
            speed_kmh = self._calculate_partial_speed_if_ready(state)
        if speed_kmh is not None:
            state["speed_kmh"] = speed_kmh
            if state.get("speed_status") not in ("medida", "medida parcial"):
                state["speed_status"] = "medida"

        detection["speed_kmh"] = state.get("speed_kmh")
        detection["speed_status"] = state.get("speed_status", "esperando cruce")
        state["seen_at"] = time.monotonic()
        self._prune()
        return detection

    def lines_for_frame(self, frame_shape):
        lines = self._line_pixels(frame_shape)
        return [
            {
                "name": "A",
                "x1": lines["A"]["x1"],
                "y1": lines["A"]["y1"],
                "x2": lines["A"]["x2"],
                "y2": lines["A"]["y2"],
                "label": "Linea A",
            },
            {
                "name": "B",
                "x1": lines["B"]["x1"],
                "y1": lines["B"]["y1"],
                "x2": lines["B"]["x2"],
                "y2": lines["B"]["y2"],
                "label": "Linea B",
            },
        ]

    def get_config(self):
        return {
            "line_a_left_y": self.line_a_left_y,
            "line_a_right_y": self.line_a_right_y,
            "line_b_left_y": self.line_b_left_y,
            "line_b_right_y": self.line_b_right_y,
            "roi_x1": self.roi_x1,
            "roi_x2": self.roi_x2,
            "distance_meters": self.distance_meters,
            "direction": self.direction,
            "hysteresis_px": self.hysteresis_px,
            "min_travel_time": self.min_travel_time,
            "max_travel_time": self.max_travel_time,
            "min_movement_px": self.min_movement_px,
            "min_partial_progress": self.min_partial_progress,
        }

    def update_config(self, **values):
        if "line_a_left_y" in values:
            self.line_a_left_y = self._clamp_normalized(values["line_a_left_y"])
        if "line_a_right_y" in values:
            self.line_a_right_y = self._clamp_normalized(values["line_a_right_y"])
        if "line_b_left_y" in values:
            self.line_b_left_y = self._clamp_normalized(values["line_b_left_y"])
        if "line_b_right_y" in values:
            self.line_b_right_y = self._clamp_normalized(values["line_b_right_y"])
        if "roi_x1" in values:
            self.roi_x1 = self._clamp_normalized(values["roi_x1"])
        if "roi_x2" in values:
            self.roi_x2 = self._clamp_normalized(values["roi_x2"])
        if self.roi_x2 < self.roi_x1:
            self.roi_x1, self.roi_x2 = self.roi_x2, self.roi_x1
        if "distance_meters" in values:
            self.distance_meters = max(0.1, float(values["distance_meters"]))
        if "direction" in values:
            self.direction = str(values["direction"]).strip().lower() or "both"
        if "hysteresis_px" in values:
            self.hysteresis_px = max(1.0, float(values["hysteresis_px"]))
        if "min_travel_time" in values:
            self.min_travel_time = max(0.01, float(values["min_travel_time"]))
            self.max_travel_time = max(self.min_travel_time, self.max_travel_time)
        if "max_travel_time" in values:
            self.max_travel_time = max(self.min_travel_time, float(values["max_travel_time"]))
        if "min_movement_px" in values:
            self.min_movement_px = max(0.0, float(values["min_movement_px"]))
        if "min_partial_progress" in values:
            self.min_partial_progress = max(0.05, min(1.0, float(values["min_partial_progress"])))
        self.reset()
        return self.get_config()

    def reset(self):
        self.history.clear()

    def _new_state(self, anchor, timestamp, lines, progress):
        return {
            "previous_anchor": anchor,
            "previous_timestamp": timestamp,
            "start_progress": progress,
            "start_progress_timestamp": timestamp,
            "min_progress": progress,
            "max_progress": progress,
            "total_movement_px": 0.0,
            "sides": {
                name: self._side(self._signed_vertical_distance(anchor, line))
                for name, line in self._measurement_lines(lines).items()
            },
            "crossings": [],
            "speed_kmh": None,
            "speed_status": self._waiting_status(progress),
            "seen_at": time.monotonic(),
        }

    def _update_motion(self, state, anchor, timestamp):
        previous_anchor = state["previous_anchor"]
        dx = anchor[0] - previous_anchor[0]
        dy = anchor[1] - previous_anchor[1]
        state["total_movement_px"] += (dx * dx + dy * dy) ** 0.5

    def _update_progress(self, state, progress, timestamp):
        state["current_progress"] = progress
        state["current_timestamp"] = timestamp
        state["min_progress"] = min(state["min_progress"], progress)
        state["max_progress"] = max(state["max_progress"], progress)
        if state["total_movement_px"] < self.min_movement_px:
            state["speed_status"] = self._waiting_status(progress)
        else:
            state["speed_status"] = "midiendo tramo"

    def _update_line_crossing(self, state, line_name, line, anchor, timestamp):
        current_signed_distance = self._signed_vertical_distance(anchor, line)
        current_side = self._side(current_signed_distance)
        previous_side = state["sides"].get(line_name)

        if current_side == 0:
            return

        if previous_side in (-1, 1) and previous_side != current_side:
            crossing_time = self._interpolated_crossing_time(
                state,
                line,
                current_signed_distance,
                anchor,
                timestamp,
            )
            self._record_crossing(state, line_name, crossing_time)

        state["sides"][line_name] = current_side

    def _record_crossing(self, state, line_name, crossing_time):
        crossings = state["crossings"]
        if crossings and crossings[-1]["line"] == line_name:
            return

        crossings.append({"line": line_name, "timestamp": crossing_time})
        if len(crossings) > 4:
            del crossings[:-4]
        state["speed_status"] = "cruce parcial"

    def _calculate_speed_if_ready(self, state):
        if state.get("speed_kmh") is not None and state.get("speed_status") == "medida":
            return state["speed_kmh"]

        crossing_a = None
        crossing_b = None
        for crossing in state["crossings"]:
            if crossing["line"] == "A":
                crossing_a = crossing
            elif crossing["line"] == "B":
                crossing_b = crossing

        if crossing_a is None or crossing_b is None:
            return None

        order = ("A", "B") if crossing_a["timestamp"] <= crossing_b["timestamp"] else ("B", "A")
        if not self._direction_allowed(order):
            state["speed_status"] = "direccion ignorada"
            return None

        dt = abs(crossing_b["timestamp"] - crossing_a["timestamp"])
        if dt < self.min_travel_time or dt > self.max_travel_time:
            state["speed_status"] = "tiempo fuera de rango"
            return None

        if state["total_movement_px"] < self.min_movement_px:
            state["speed_status"] = "sin movimiento real"
            return None

        state["speed_status"] = "medida"
        return (self.distance_meters / dt) * 3.6

    def _calculate_partial_speed_if_ready(self, state):
        if state.get("speed_kmh") is not None:
            state["speed_status"] = "medida parcial"
            return state["speed_kmh"]

        progress_delta = state["max_progress"] - state["min_progress"]
        if progress_delta < self.min_partial_progress:
            if state["total_movement_px"] >= self.min_movement_px:
                state["speed_status"] = "midiendo tramo"
            return None

        dt = abs(state.get("current_timestamp", state["previous_timestamp"]) - state["start_progress_timestamp"])
        if dt < self.min_travel_time or dt > self.max_travel_time:
            state["speed_status"] = "tiempo fuera de rango"
            return None

        if state["total_movement_px"] < self.min_movement_px:
            state["speed_status"] = "sin movimiento real"
            return None

        state["speed_status"] = "medida parcial"
        distance = self.distance_meters * min(1.0, progress_delta)
        return (distance / dt) * 3.6

    def _direction_allowed(self, order):
        if self.direction == "both":
            return True

        top_line = "A" if self._line_average_y("A") < self._line_average_y("B") else "B"
        bottom_line = "B" if top_line == "A" else "A"
        if self.direction == "down":
            return order == (top_line, bottom_line)
        if self.direction == "up":
            return order == (bottom_line, top_line)
        return True

    def _interpolated_crossing_time(self, state, line, current_signed_distance, anchor, timestamp):
        previous_anchor = state["previous_anchor"]
        previous_timestamp = state["previous_timestamp"]
        previous_signed_distance = self._signed_vertical_distance(previous_anchor, line)
        denominator = previous_signed_distance - current_signed_distance
        if denominator == 0:
            return timestamp

        ratio = previous_signed_distance / denominator
        ratio = max(0.0, min(1.0, ratio))
        return previous_timestamp + (timestamp - previous_timestamp) * ratio

    def _line_pixels(self, frame_shape):
        height, width = frame_shape[:2]
        x1 = self.roi_x1 * width
        x2 = self.roi_x2 * width
        return {
            "A": {
                "x1": x1,
                "y1": self.line_a_left_y * height,
                "x2": x2,
                "y2": self.line_a_right_y * height,
            },
            "B": {
                "x1": x1,
                "y1": self.line_b_left_y * height,
                "x2": x2,
                "y2": self.line_b_right_y * height,
            },
        }

    def _progress_between_lines(self, anchor, lines):
        line_a_y = self._line_y_at_x(anchor[0], lines["A"])
        line_b_y = self._line_y_at_x(anchor[0], lines["B"])
        denominator = line_b_y - line_a_y
        if denominator == 0:
            return 0.0
        return (anchor[1] - line_a_y) / denominator

    def _waiting_status(self, progress):
        if progress < 0:
            return "antes de linea A"
        if progress > 1:
            return "despues de linea B"
        return "entre lineas"

    def _inside_roi(self, anchor, frame_shape):
        width = frame_shape[1]
        x1 = self.roi_x1 * width
        x2 = self.roi_x2 * width
        return x1 <= anchor[0] <= x2

    @staticmethod
    def _measurement_lines(lines):
        return {"A": lines["A"], "B": lines["B"]}

    def _side(self, signed_distance):
        if signed_distance < -self.hysteresis_px:
            return -1
        if signed_distance > self.hysteresis_px:
            return 1
        return 0

    @staticmethod
    def _signed_vertical_distance(anchor, line):
        return anchor[1] - SpeedEstimator._line_y_at_x(anchor[0], line)

    @staticmethod
    def _line_y_at_x(x, line):
        x1 = line["x1"]
        x2 = line["x2"]
        if x1 == x2:
            return (line["y1"] + line["y2"]) / 2

        ratio = (x - x1) / (x2 - x1)
        ratio = max(0.0, min(1.0, ratio))
        return line["y1"] + (line["y2"] - line["y1"]) * ratio

    def _line_average_y(self, name):
        if name == "A":
            return (self.line_a_left_y + self.line_a_right_y) / 2
        return (self.line_b_left_y + self.line_b_right_y) / 2

    @staticmethod
    def _anchor(detection):
        return (
            (float(detection["x1"]) + float(detection["x2"])) / 2,
            (float(detection["y1"]) + float(detection["y2"])) / 2,
        )

    @staticmethod
    def _clamp_normalized(value):
        return max(0.0, min(1.0, float(value)))

    def _prune(self):
        now = time.monotonic()
        expired = [
            track_id
            for track_id, item in self.history.items()
            if now - item["seen_at"] > self.max_history_age
        ]
        for track_id in expired:
            self.history.pop(track_id, None)
