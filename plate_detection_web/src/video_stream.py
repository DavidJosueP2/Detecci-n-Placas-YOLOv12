import time
from threading import Lock

import cv2

from src.frame_processor import (
    encode_jpeg,
    placeholder_crop,
    placeholder_frame,
    process_frame,
)
from src.speed_estimator import SpeedEstimator
from src.utils import detection_status, now_label


class VideoStream:
    def __init__(
        self,
        detector,
        plate_reader=None,
        source=0,
        pixels_per_meter=45.0,
        speed_smoothing=0.35,
        detector_error=None,
    ):
        self.detector = detector
        self.plate_reader = plate_reader
        self.source = source
        self.source_label = str(source)
        self.source_version = 0
        self.detector_error = detector_error
        self.lock = Lock()
        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=pixels_per_meter,
            smoothing=speed_smoothing,
        )
        self.plate_memory = {}
        self.last_crop = encode_jpeg(placeholder_crop())
        self.crop_slots = [self.last_crop]
        self.last_frame = encode_jpeg(placeholder_frame("Esperando video"))
        self.paused = False
        self.seek_request_seconds = None
        self.position_seconds = 0.0
        self.duration_seconds = 0.0
        self.is_seekable = isinstance(source, str)
        self.status = {
            "detected": False,
            "confidence": 0.0,
            "class_id": None,
            "class_name": None,
            "track_id": None,
            "detections": [],
            "plate_text": "",
            "plate_text_confidence": 0.0,
            "speed_kmh": None,
            "paused": self.paused,
            "position_seconds": self.position_seconds,
            "duration_seconds": self.duration_seconds,
            "is_seekable": self.is_seekable,
            "message": detector_error or "Sin deteccion",
            "source": self.source_label,
            "timestamp": now_label(),
        }

    def frames(self):
        if self.detector is None:
            yield self._multipart_frame(placeholder_frame(self.detector_error))
            return

        capture, capture_version, source = self._open_capture()
        self._update_capture_info(capture, source)
        self._restore_video_position(capture, source)
        if not capture.isOpened():
            message = f"No se pudo abrir la fuente de video: {source}"
            with self.lock:
                self.status.update(
                    {
                        "detected": False,
                        "message": message,
                        "timestamp": now_label(),
                    }
                )
            while True:
                yield self._multipart_frame(placeholder_frame(message))
                time.sleep(1)

        while True:
            with self.lock:
                source_changed = capture_version != self.source_version

            if source_changed:
                capture.release()
                capture, capture_version, source = self._open_capture()
                self._update_capture_info(capture, source)
                self._restore_video_position(capture, source)
                if not capture.isOpened():
                    message = f"No se pudo abrir la fuente de video: {source}"
                    yield self._multipart_frame(placeholder_frame(message))
                    time.sleep(1)
                    continue

            with self.lock:
                paused = self.paused
                paused_frame = self.last_frame
                seek_request_seconds = self.seek_request_seconds
                self.seek_request_seconds = None

            if seek_request_seconds is not None and isinstance(source, str):
                seek_to = max(0.0, min(seek_request_seconds, self.duration_seconds or seek_request_seconds))
                capture.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000)
                self.speed_estimator.reset()
                paused = False

            if paused:
                if paused_frame is not None:
                    yield self._multipart_payload(paused_frame)
                capture.release()
                return

            ok, frame = capture.read()
            if not ok:
                if isinstance(source, str):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                time.sleep(0.1)
                continue

            timestamp = self._frame_timestamp(capture, source)
            display_position = timestamp if isinstance(source, str) else 0.0
            with self.lock:
                self.position_seconds = display_position
            detections = self.detector.track(frame)
            detections = self._enrich_detections(frame, detections, timestamp)
            processed, crops, best_detection = process_frame(frame, detections)
            frame_bytes = encode_jpeg(processed)

            with self.lock:
                self.status = detection_status(best_detection)
                self.status["source"] = self.source_label
                self.status["paused"] = self.paused
                self.status["position_seconds"] = self.position_seconds
                self.status["duration_seconds"] = self.duration_seconds
                self.status["is_seekable"] = self.is_seekable
                self.status["detections"] = [
                    self._public_detection(item["detection"], index)
                    for index, item in enumerate(crops[:6])
                ]
                if crops:
                    encoded_crops = [encode_jpeg(item["crop"]) for item in crops[:6]]
                    self.crop_slots = [crop for crop in encoded_crops if crop is not None]
                    self.last_crop = self.crop_slots[0] if self.crop_slots else encode_jpeg(placeholder_crop())
                else:
                    self.crop_slots = [encode_jpeg(placeholder_crop())]
                    self.last_crop = self.crop_slots[0]
                if frame_bytes is not None:
                    self.last_frame = frame_bytes

            if frame_bytes is not None:
                yield self._multipart_payload(frame_bytes)

    def set_source(self, source, source_label=None):
        with self.lock:
            self.source = source
            self.source_label = source_label or str(source)
            self.source_version += 1
            self.last_crop = encode_jpeg(placeholder_crop())
            self.crop_slots = [self.last_crop]
            self.last_frame = encode_jpeg(placeholder_frame("Fuente actualizada"))
            self.paused = False
            self.seek_request_seconds = None
            self.position_seconds = 0.0
            self.duration_seconds = 0.0
            self.is_seekable = isinstance(source, str)
            self.speed_estimator.reset()
            self.plate_memory.clear()
            self.status.update(
                {
                    "detected": False,
                    "confidence": 0.0,
                    "class_id": None,
                    "class_name": None,
                    "track_id": None,
                    "detections": [],
                    "plate_text": "",
                    "plate_text_confidence": 0.0,
                    "speed_kmh": None,
                    "paused": self.paused,
                    "position_seconds": self.position_seconds,
                    "duration_seconds": self.duration_seconds,
                    "is_seekable": self.is_seekable,
                    "message": "Fuente de video actualizada",
                    "source": self.source_label,
                    "timestamp": now_label(),
                }
            )

    def current_crop(self):
        with self.lock:
            return self.last_crop or encode_jpeg(placeholder_crop())

    def current_crop_at(self, index):
        with self.lock:
            if 0 <= index < len(self.crop_slots):
                return self.crop_slots[index]
            return encode_jpeg(placeholder_crop())

    def current_frame(self):
        with self.lock:
            return self.last_frame or encode_jpeg(placeholder_frame("Esperando video"))

    def current_status(self):
        with self.lock:
            return dict(self.status)

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            self.status["paused"] = self.paused
            self.status["message"] = "Video pausado" if self.paused else "Video reproduciendo"
            self.status["timestamp"] = now_label()
            return self.paused

    def set_paused(self, paused):
        with self.lock:
            self.paused = bool(paused)
            self.status["paused"] = self.paused
            self.status["message"] = "Video pausado" if self.paused else "Video reproduciendo"
            self.status["timestamp"] = now_label()
            return self.paused

    def seek_relative(self, delta_seconds):
        with self.lock:
            if not isinstance(self.source, str):
                return self.position_seconds
            target = self.position_seconds + float(delta_seconds)
            self.seek_request_seconds = max(0.0, target)
            self.status["message"] = "Buscando posicion"
            self.status["timestamp"] = now_label()
            return self.seek_request_seconds

    def seek_to(self, seconds):
        with self.lock:
            if not isinstance(self.source, str):
                return self.position_seconds
            target = max(0.0, float(seconds))
            if self.duration_seconds:
                target = min(target, self.duration_seconds)
            self.seek_request_seconds = target
            self.status["message"] = "Buscando posicion"
            self.status["timestamp"] = now_label()
            return self.seek_request_seconds

    def _open_capture(self):
        with self.lock:
            source = self.source
            version = self.source_version
        return cv2.VideoCapture(source), version, source

    def _update_capture_info(self, capture, source):
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        with self.lock:
            self.duration_seconds = duration if isinstance(source, str) else 0.0
            self.is_seekable = isinstance(source, str) and duration > 0
            self.status["duration_seconds"] = self.duration_seconds
            self.status["is_seekable"] = self.is_seekable

    def _restore_video_position(self, capture, source):
        if not isinstance(source, str):
            return
        with self.lock:
            position = self.position_seconds
        if position > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000)

    def _frame_timestamp(self, capture, source):
        if isinstance(source, str):
            video_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if video_seconds > 0:
                return video_seconds
        return time.monotonic()

    def _enrich_detections(self, frame, detections, timestamp):
        enriched = []
        for detection in detections:
            self.speed_estimator.update(detection, timestamp=timestamp)
            crop = self._crop_detection(frame, detection)

            plate_text = ""
            plate_text_confidence = 0.0
            plate_chars = []
            if self.plate_reader is not None and crop is not None:
                plate_text, plate_text_confidence, plate_chars = self.plate_reader.read(crop)

            plate_text, plate_text_confidence = self._stabilize_plate_text(
                detection,
                plate_text,
                plate_text_confidence,
            )
            detection["plate_text"] = plate_text
            detection["plate_text_confidence"] = plate_text_confidence
            detection["characters"] = [
                {"value": char["value"], "confidence": char["confidence"]}
                for char in plate_chars
            ]
            enriched.append(detection)

        return enriched

    def _stabilize_plate_text(self, detection, text, confidence):
        track_id = detection.get("track_id")
        if track_id is None:
            return text, confidence

        previous = self.plate_memory.get(track_id)
        clean_text = "".join(char for char in text if char.isalnum()).upper()

        if clean_text and (
            previous is None
            or len(clean_text) > len(previous["text"])
            or confidence >= previous["confidence"]
        ):
            self.plate_memory[track_id] = {
                "text": clean_text,
                "confidence": confidence,
                "seen_at": time.monotonic(),
            }
            return clean_text, confidence

        if previous is not None and time.monotonic() - previous["seen_at"] < 5:
            return previous["text"], previous["confidence"]

        return clean_text, confidence

    @staticmethod
    def _crop_detection(frame, detection):
        height, width = frame.shape[:2]
        x1 = max(0, min(int(detection["x1"]), width - 1))
        y1 = max(0, min(int(detection["y1"]), height - 1))
        x2 = max(0, min(int(detection["x2"]), width - 1))
        y2 = max(0, min(int(detection["y2"]), height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _multipart_frame(self, frame):
        frame_bytes = encode_jpeg(frame)
        return self._multipart_payload(frame_bytes)

    @staticmethod
    def _public_detection(detection, index):
        return {
            "index": index,
            "confidence": detection["confidence"],
            "class_id": detection["class_id"],
            "class_name": "License Plate",
            "track_id": detection.get("track_id"),
            "plate_text": detection.get("plate_text") or "",
            "plate_text_confidence": detection.get("plate_text_confidence", 0.0),
            "characters": detection.get("characters", []),
            "speed_kmh": detection.get("speed_kmh"),
        }

    @staticmethod
    def _multipart_payload(frame_bytes):
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
