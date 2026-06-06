import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2

from src.activity_tracker import ActivityTracker
from src.crop_quality import plate_crop_cut_risk, plate_crop_ghost_risk, plate_crop_quality
from src.frame_processor import (
    encode_jpeg,
    placeholder_crop,
    placeholder_frame,
    process_frame,
    resize_to_max_width,
)
from src.lightweight_tracker import LightweightTracker
from src.live_frame_reader import LiveFrameReader
from src.plate_memory import PlateMemory
from src.plate_postprocess import postprocess_ecuador_plate
from src.speed_estimator import SpeedEstimator
from src.utils import detection_status, now_label


class VideoStream:
    def __init__(
        self,
        detector,
        plate_reader=None,
        source=0,
        speed_line_a_y=0.45,
        speed_line_b_y=0.70,
        speed_line_a_right_y=None,
        speed_line_b_right_y=None,
        speed_roi_x1=0.0,
        speed_roi_x2=1.0,
        speed_distance_meters=5.0,
        speed_direction="both",
        speed_line_hysteresis_px=8.0,
        speed_min_travel_time=0.15,
        speed_max_travel_time=8.0,
        speed_min_movement_px=35.0,
        speed_min_partial_progress=0.35,
        target_fps=12.0,
        stream_max_width=960,
        stream_jpeg_quality=68,
        detection_every_n_frames=6,
        live_detection_interval_seconds=0.12,
        live_detection_mode="detect",
        ocr_interval_seconds=1.25,
        ocr_retry_interval_seconds=0.30,
        ocr_max_plates_per_frame=1,
        ocr_min_detection_confidence=0.20,
        ocr_zone_x1=0.0,
        ocr_zone_y1=0.0,
        ocr_zone_x2=1.0,
        ocr_zone_y2=1.0,
        ocr_zone_min_overlap=0.20,
        camera_backend="msmf",
        camera_width=640,
        camera_height=480,
        camera_fps=30.0,
        camera_fourcc="MJPG",
        incident_service=None,
        detector_error=None,
    ):
        self.detector = detector
        self.plate_reader = plate_reader
        self.source = source
        self.source_label = str(source)
        self.source_version = 0
        self.detector_error = detector_error
        self.target_fps = max(1.0, float(target_fps))
        self.stream_max_width = max(0, int(stream_max_width))
        self.stream_jpeg_quality = max(35, min(95, int(stream_jpeg_quality)))
        self.detection_every_n_frames = max(1, int(detection_every_n_frames))
        self.live_detection_interval_seconds = max(0.0, float(live_detection_interval_seconds))
        self.live_detection_mode = str(live_detection_mode or "detect").strip().lower()
        if self.live_detection_mode not in {"detect", "track"}:
            self.live_detection_mode = "detect"
        self.ocr_interval_seconds = max(0.0, float(ocr_interval_seconds))
        self.ocr_retry_interval_seconds = max(0.0, float(ocr_retry_interval_seconds))
        self.ocr_max_plates_per_frame = max(0, int(ocr_max_plates_per_frame))
        self.ocr_min_detection_confidence = max(0.0, float(ocr_min_detection_confidence))
        self.ocr_zone = self._normalize_ocr_zone(
            ocr_zone_x1,
            ocr_zone_y1,
            ocr_zone_x2,
            ocr_zone_y2,
        )
        self.ocr_zone_min_overlap = max(0.0, min(1.0, float(ocr_zone_min_overlap)))
        self.camera_backend = str(camera_backend or "auto").strip().lower()
        self.camera_width = max(0, int(camera_width))
        self.camera_height = max(0, int(camera_height))
        self.camera_fps = max(0.0, float(camera_fps))
        self.camera_fourcc = str(camera_fourcc or "").strip().upper()
        self.incident_service = incident_service
        self.source_fps = 0.0
        self.display_fps = 0.0
        self.detection_fps = 0.0
        self.detector_ms = 0.0
        self.capture_width = 0
        self.capture_height = 0
        self.capture_backend = ""
        self.last_live_detection_submitted_at = 0.0
        self.last_detection_completed_at = 0.0
        self.last_frame_started_at = 0.0
        self.lock = Lock()
        self.speed_estimator = SpeedEstimator(
            line_a_y=speed_line_a_y,
            line_b_y=speed_line_b_y,
            line_a_right_y=speed_line_a_right_y,
            line_b_right_y=speed_line_b_right_y,
            roi_x1=speed_roi_x1,
            roi_x2=speed_roi_x2,
            distance_meters=speed_distance_meters,
            direction=speed_direction,
            hysteresis_px=speed_line_hysteresis_px,
            min_travel_time=speed_min_travel_time,
            max_travel_time=speed_max_travel_time,
            min_movement_px=speed_min_movement_px,
            min_partial_progress=speed_min_partial_progress,
        )
        self.plate_memory = PlateMemory()
        self.activity_tracker = ActivityTracker(
            incident_service=incident_service,
            speed_estimator=self.speed_estimator,
            ocr_zone_fn=self.ocr_zone_for_frame,
            plate_reader=plate_reader,
            debug_root=incident_service.activity_root if incident_service is not None else None,
        )
        self.last_detections = []
        self.tracker = LightweightTracker()
        self.ocr_executor = ThreadPoolExecutor(max_workers=1)
        self.detector_executor = ThreadPoolExecutor(max_workers=1)
        self.pending_detection = None
        self.pending_detection_version = None
        self.pending_ocr = {}
        self.last_ocr_attempts = {}
        self.last_ocr_attempt_at = 0.0
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
        live_reader = self._live_reader_for_source(capture, source)
        live_frame_version = 0
        last_emit_time = 0.0
        frame_index = 0
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
            frame_started_at = time.monotonic()
            with self.lock:
                source_changed = capture_version != self.source_version

            if source_changed:
                self.activity_tracker.flush()
                self._close_capture(capture, live_reader)
                capture, capture_version, source = self._open_capture()
                self._update_capture_info(capture, source)
                self._restore_video_position(capture, source)
                live_reader = self._live_reader_for_source(capture, source)
                live_frame_version = 0
                last_emit_time = 0.0
                frame_index = 0
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
                self.activity_tracker.flush()
                self.speed_estimator.reset()
                self.last_ocr_attempts.clear()
                self.last_ocr_attempt_at = 0.0
                self.last_detections = []
                self.tracker.clear()
                self.pending_ocr.clear()
                last_emit_time = 0.0
                frame_index = 0
                paused = False

            if paused:
                if paused_frame is not None:
                    yield self._multipart_payload(paused_frame)
                self.activity_tracker.flush()
                self._close_capture(capture, live_reader)
                return

            if live_reader is not None:
                ok, frame, timestamp, live_frame_version = live_reader.read(live_frame_version)
            else:
                ok, frame = capture.read()
                timestamp = self._frame_timestamp(capture, source) if ok else 0.0
            if not ok:
                if isinstance(source, str):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.activity_tracker.flush()
                    self.speed_estimator.reset()
                    self.plate_memory.clear()
                    self.last_ocr_attempts.clear()
                    self.last_ocr_attempt_at = 0.0
                    self.last_detections = []
                    self.tracker.clear()
                    self.pending_ocr.clear()
                    last_emit_time = 0.0
                    frame_index = 0
                    time.sleep(0.1)
                    continue
                time.sleep(0.1)
                continue

            self._update_frame_capture_info(frame, source)
            display_position = timestamp if isinstance(source, str) else 0.0
            with self.lock:
                self.position_seconds = display_position

            if isinstance(source, str):
                run_detection = self._should_detect_this_frame(source, frame_index)
            else:
                self._collect_async_detection(capture_version)
                self._submit_async_detection(frame, source, timestamp, capture_version)
                run_detection = False

            if run_detection:
                detector_started_at = time.monotonic()
                detections = self._detect_frame(frame, source, timestamp)
                self.detector_ms = (time.monotonic() - detector_started_at) * 1000
                detections = self._enrich_detections(frame, detections, timestamp)
                self.last_detections = detections
                self._mark_detection_completed()
            else:
                if isinstance(source, str):
                    self._collect_ocr_results()
                detections = [dict(item) for item in self.last_detections]
                for detection in detections:
                    self._apply_cached_plate_text(detection)
            speed_lines = self.speed_estimator.lines_for_frame(frame.shape)
            stats_detection = self._best_detection_for_stats(detections)
            processed, crops, best_detection = process_frame(
                frame,
                detections,
                speed_lines=speed_lines,
                ocr_zone=self.ocr_zone_for_frame(frame.shape),
                stats=self._frame_stats(detections, stats_detection),
            )
            output_frame = resize_to_max_width(processed, self.stream_max_width)
            frame_bytes = encode_jpeg(output_frame, quality=self.stream_jpeg_quality)
            encoded_crop_items = self._prepare_crop_items(crops[:6], frame_bytes)

            with self.lock:
                self.status = detection_status(best_detection)
                self.status["source"] = self.source_label
                self.status["paused"] = self.paused
                self.status["position_seconds"] = self.position_seconds
                self.status["duration_seconds"] = self.duration_seconds
                self.status["is_seekable"] = self.is_seekable
                self.status["stats"] = self._frame_stats(detections, best_detection)
                self.status["detections"] = [
                    self._public_detection(item["detection"], index)
                    for index, item in enumerate(encoded_crop_items)
                ]
                if encoded_crop_items:
                    self.crop_slots = [item["bytes"] for item in encoded_crop_items]
                    self.last_crop = self.crop_slots[0] if self.crop_slots else encode_jpeg(placeholder_crop())
                else:
                    self.crop_slots = [encode_jpeg(placeholder_crop())]
                    self.last_crop = self.crop_slots[0]
                if frame_bytes is not None:
                    self.last_frame = frame_bytes

            if frame_bytes is not None:
                last_emit_time = self._pace_video_file(source, last_emit_time)
                self._update_display_fps(frame_started_at)
                yield self._multipart_payload(frame_bytes)
            frame_index += 1

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
            self.activity_tracker.flush()
            self.speed_estimator.reset()
            self.plate_memory.clear()
            self.last_detections = []
            self.tracker.clear()
            self.pending_ocr.clear()
            self.pending_detection = None
            self.pending_detection_version = None
            self.detection_fps = 0.0
            self.detector_ms = 0.0
            self.last_detection_completed_at = 0.0
            self.last_ocr_attempts.clear()
            self.last_ocr_attempt_at = 0.0
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
                    "speed_status": "esperando cruce",
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

    def current_speed_config(self):
        with self.lock:
            return self.speed_estimator.get_config()

    def current_ocr_zone_config(self):
        with self.lock:
            return dict(self.ocr_zone)

    def update_speed_config(self, **values):
        with self.lock:
            config = self.speed_estimator.update_config(**values)
            self.status["message"] = "Configuracion de velocidad actualizada"
            self.status["timestamp"] = now_label()
            return config

    def update_ocr_zone_config(self, **values):
        with self.lock:
            current = dict(self.ocr_zone)
            current.update(values)
            self.ocr_zone = self._normalize_ocr_zone(
                current.get("x1", 0.0),
                current.get("y1", 0.0),
                current.get("x2", 1.0),
                current.get("y2", 1.0),
            )
            self.status["message"] = "Zona OCR actualizada"
            self.status["timestamp"] = now_label()
            return dict(self.ocr_zone)

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
        if isinstance(source, int):
            capture = self._open_camera_capture(source)
            self._configure_camera_capture(capture)
        else:
            capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture, version, source

    @staticmethod
    def _live_reader_for_source(capture, source):
        if isinstance(source, str) or not capture.isOpened():
            return None
        return LiveFrameReader(capture)

    @staticmethod
    def _close_capture(capture, live_reader=None):
        if live_reader is not None:
            live_reader.close()
            return
        capture.release()

    def _open_camera_capture(self, source):
        tried = set()
        for backend in self._camera_backend_candidates():
            if backend in tried:
                continue
            tried.add(backend)
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(source)

    def _camera_backend_candidates(self):
        preferred = self._camera_backend_value()
        candidates = [preferred, cv2.CAP_MSMF, cv2.CAP_ANY]
        if self.camera_backend in {"dshow", "directshow"}:
            candidates.append(cv2.CAP_DSHOW)
        return candidates

    def _camera_backend_value(self):
        backends = {
            "dshow": cv2.CAP_DSHOW,
            "directshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "any": cv2.CAP_ANY,
            "auto": cv2.CAP_ANY,
        }
        return backends.get(self.camera_backend, cv2.CAP_MSMF)

    def _configure_camera_capture(self, capture):
        if self.camera_fourcc and len(self.camera_fourcc) == 4:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.camera_fourcc),
            )
        if self.camera_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        if self.camera_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.camera_fps)

    def _update_capture_info(self, capture, source):
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        with self.lock:
            self.source_fps = fps if fps > 0 else 0.0
            self.capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.capture_backend = self._capture_backend_name(capture)
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
            fps = self.source_fps or 0.0
            frame_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if fps > 0 and frame_index > 0:
                return frame_index / fps
        return time.monotonic()

    def _target_interval_for_source(self, source):
        if not isinstance(source, str):
            return 0.0

        fps = self.source_fps or self.target_fps
        return 1.0 / max(1.0, fps)

    def _pace_video_file(self, source, last_emit_time):
        if not isinstance(source, str):
            return time.monotonic()

        interval = self._target_interval_for_source(source)
        if last_emit_time > 0 and interval > 0:
            wait_seconds = (last_emit_time + interval) - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)

        return time.monotonic()

    def _update_display_fps(self, frame_started_at):
        elapsed = max(0.001, time.monotonic() - frame_started_at)
        fps = 1.0 / elapsed
        self.display_fps = fps if self.display_fps <= 0 else self.display_fps * 0.8 + fps * 0.2

    def _update_frame_capture_info(self, frame, source):
        height, width = frame.shape[:2]
        with self.lock:
            self.capture_width = width
            self.capture_height = height
            self.is_seekable = isinstance(source, str) and self.duration_seconds > 0

    @staticmethod
    def _capture_backend_name(capture):
        try:
            return capture.getBackendName()
        except Exception:
            return ""

    def _mark_detection_completed(self):
        now = time.monotonic()
        if self.last_detection_completed_at > 0:
            elapsed = max(0.001, now - self.last_detection_completed_at)
            fps = 1.0 / elapsed
            self.detection_fps = fps if self.detection_fps <= 0 else self.detection_fps * 0.75 + fps * 0.25
        self.last_detection_completed_at = now

    def _frame_stats(self, detections, best_detection):
        speed_status = None
        if best_detection is not None:
            speed = best_detection.get("speed_kmh")
            speed_status = (
                f"{speed:.1f} km/h"
                if speed is not None
                else best_detection.get("speed_status", "sin velocidad")
            )
        return {
            "fps": self.display_fps,
            "source_fps": self.source_fps,
            "detection_fps": self.detection_fps,
            "detector_ms": self.detector_ms,
            "detections": len(detections or []),
            "tracks": self._lightweight_track_count(),
            "speed_status": speed_status,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "capture_backend": self.capture_backend,
            "detection_mode": self._detection_mode_label(),
            "live_interval_seconds": self.live_detection_interval_seconds,
            "stream_width": self.stream_max_width,
            "stream_jpeg_quality": self.stream_jpeg_quality,
        }

    def _lightweight_track_count(self):
        return self.tracker.count()

    def _detection_mode_label(self):
        if self.live_detection_mode == "track":
            return "track"
        return "detect"

    @staticmethod
    def _best_detection_for_stats(detections):
        if not detections:
            return None
        tracked = [item for item in detections if item.get("track_id") is not None]
        candidates = tracked or detections
        return max(candidates, key=lambda item: item.get("confidence", 0.0))

    def _should_detect_this_frame(self, source, frame_index):
        if not isinstance(source, str):
            return True

        return frame_index % self.detection_every_n_frames == 0

    def _submit_async_detection(self, frame, source, timestamp, capture_version):
        now = time.monotonic()
        if now - self.last_live_detection_submitted_at < self.live_detection_interval_seconds:
            return

        if self.pending_detection is not None and not self.pending_detection.done():
            return

        if self.pending_detection is not None:
            self._collect_async_detection(capture_version)
            if self.pending_detection is not None:
                return

        self.pending_detection_version = capture_version
        self.last_live_detection_submitted_at = now
        self.pending_detection = self.detector_executor.submit(
            self._detect_and_enrich_frame,
            frame.copy(),
            source,
            timestamp,
            capture_version,
        )

    def _collect_async_detection(self, capture_version):
        if self.pending_detection is None or not self.pending_detection.done():
            return

        future = self.pending_detection
        future_version = self.pending_detection_version
        self.pending_detection = None
        self.pending_detection_version = None

        if future_version != capture_version:
            return

        try:
            detections, detector_ms = future.result()
        except Exception:
            return

        self.detector_ms = detector_ms
        self.last_detections = detections
        self._mark_detection_completed()

    def _detect_and_enrich_frame(self, frame, source, timestamp, capture_version):
        detector_started_at = time.monotonic()
        detections = self._detect_raw_frame(frame, source)
        detector_ms = (time.monotonic() - detector_started_at) * 1000

        if not self._is_capture_version_current(capture_version):
            return [], detector_ms

        if self._should_assign_lightweight_tracks(source):
            detections = self._assign_lightweight_track_ids(detections, timestamp)
        detections = self._enrich_detections(frame, detections, timestamp)
        return detections, detector_ms

    def _detect_frame(self, frame, source, timestamp):
        detections = self._detect_raw_frame(frame, source)
        if self._should_assign_lightweight_tracks(source):
            return self._assign_lightweight_track_ids(detections, timestamp)
        return detections

    def _detect_raw_frame(self, frame, source):
        if not isinstance(source, str) and self.live_detection_mode == "track":
            return self.detector.track(frame)
        return self.detector.detect(frame)

    def _should_assign_lightweight_tracks(self, source):
        return isinstance(source, str) or self.live_detection_mode != "track"

    def _is_capture_version_current(self, capture_version):
        with self.lock:
            return capture_version == self.source_version

    def _assign_lightweight_track_ids(self, detections, timestamp):
        return self.tracker.assign(detections, timestamp)

    @staticmethod
    def _box_tuple(detection):
        return (
            float(detection["x1"]),
            float(detection["y1"]),
            float(detection["x2"]),
            float(detection["y2"]),
        )

    @staticmethod
    def _center(detection):
        return (
            (float(detection["x1"]) + float(detection["x2"])) / 2,
            (float(detection["y1"]) + float(detection["y2"])) / 2,
        )

    @staticmethod
    def _normalize_ocr_zone(x1, y1, x2, y2):
        left = max(0.0, min(1.0, float(x1)))
        top = max(0.0, min(1.0, float(y1)))
        right = max(0.0, min(1.0, float(x2)))
        bottom = max(0.0, min(1.0, float(y2)))

        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top

        min_size = 0.02
        if right - left < min_size:
            right = min(1.0, left + min_size)
            left = max(0.0, right - min_size)
        if bottom - top < min_size:
            bottom = min(1.0, top + min_size)
            top = max(0.0, bottom - min_size)

        return {"x1": left, "y1": top, "x2": right, "y2": bottom}

    def ocr_zone_for_frame(self, frame_shape):
        height, width = frame_shape[:2]
        zone = self.ocr_zone
        return {
            "x1": zone["x1"] * width,
            "y1": zone["y1"] * height,
            "x2": zone["x2"] * width,
            "y2": zone["y2"] * height,
            "label": "Zona OCR",
        }

    def _is_detection_in_ocr_zone(self, detection, frame_shape):
        zone = self.ocr_zone_for_frame(frame_shape)
        box = self._box_tuple(detection)
        zone_box = (zone["x1"], zone["y1"], zone["x2"], zone["y2"])
        center_x, center_y = self._center(detection)

        center_inside = (
            zone_box[0] <= center_x <= zone_box[2]
            and zone_box[1] <= center_y <= zone_box[3]
        )
        if center_inside:
            return True

        x1 = max(box[0], zone_box[0])
        y1 = max(box[1], zone_box[1])
        x2 = min(box[2], zone_box[2])
        y2 = min(box[3], zone_box[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        return inter / box_area >= self.ocr_zone_min_overlap

    @staticmethod
    def _is_detection_near_frame_edge(detection, frame_shape):
        height, width = frame_shape[:2]
        margin_x = max(8.0, width * 0.012)
        margin_y = max(6.0, height * 0.012)
        return (
            float(detection.get("x1", 0.0)) <= margin_x
            or float(detection.get("y1", 0.0)) <= margin_y
            or float(detection.get("x2", width)) >= width - margin_x
            or float(detection.get("y2", height)) >= height - margin_y
        )

    def _enrich_detections(self, frame, detections, timestamp):
        self._collect_ocr_results()
        enriched = []
        for detection in detections:
            detection["ocr_zone_active"] = self._is_detection_in_ocr_zone(detection, frame.shape)
            detection["_edge_clipped"] = self._is_detection_near_frame_edge(detection, frame.shape)
        ocr_candidates = self._ocr_candidate_ids(detections)
        snapshot_frame = None

        for detection in detections:
            self.speed_estimator.update(
                detection,
                timestamp=timestamp,
                frame_shape=frame.shape,
            )
            snapshot_crop = self._crop_detection(frame, detection)
            if snapshot_crop is not None:
                detection["_crop_quality"] = plate_crop_quality(snapshot_crop)
                detection["_crop_cut_risk"] = plate_crop_cut_risk(snapshot_crop)
                detection["_crop_ghost_risk"] = plate_crop_ghost_risk(snapshot_crop)
                if snapshot_frame is None:
                    snapshot_frame = frame.copy()
                detection["_snapshot_crop"] = snapshot_crop
                detection["_snapshot_frame"] = snapshot_frame
                detection["_snapshot_timestamp"] = timestamp

            plate_text = ""
            plate_text_confidence = 0.0
            plate_chars = []
            cached = self.plate_memory.get(detection)
            if cached:
                plate_text = cached["text"]
                plate_text_confidence = cached["confidence"]
                plate_chars = cached.get("characters", [])
                detection["plate_original_text"] = cached.get("original_text", "")
                detection["plate_clean_text"] = cached.get("clean_text", "")
                detection["plate_postprocess"] = cached.get("postprocess")

            if id(detection) in ocr_candidates and self._should_run_ocr(detection):
                self._submit_ocr(detection, snapshot_crop)

            plate_text, plate_text_confidence, plate_chars = self._stabilize_plate_text(
                detection,
                plate_text,
                plate_text_confidence,
                plate_chars,
            )
            detection["plate_text"] = plate_text
            detection["plate_text_confidence"] = plate_text_confidence
            detection["characters"] = [
                {"value": char["value"], "confidence": char["confidence"]}
                for char in plate_chars
            ]
            if snapshot_crop is not None:
                activity_frame = snapshot_frame if snapshot_frame is not None else frame
                self.activity_tracker.update(detection, activity_frame, snapshot_crop, timestamp, self.source_label)
            enriched.append(detection)

        self.activity_tracker.prune()
        return enriched

    def _submit_ocr(self, detection, crop):
        if crop is None or self.plate_reader is None:
            return

        key = self.plate_memory.key_for(detection)
        if key in self.pending_ocr:
            return

        self._mark_ocr_attempt(detection)
        self.pending_ocr[key] = self.ocr_executor.submit(self.plate_reader.read, crop)

    def _collect_ocr_results(self):
        completed = [
            (key, future)
            for key, future in self.pending_ocr.items()
            if future.done()
        ]
        for key, future in completed:
            self.pending_ocr.pop(key, None)
            try:
                text, confidence, characters = future.result()
            except Exception:
                continue

            postprocess = postprocess_ecuador_plate(text, confidence, characters)
            corrected_text = postprocess["texto_corregido"]
            if not corrected_text:
                continue

            previous = self.plate_memory.get_raw(key)
            if previous and len(corrected_text) < len(previous["text"]) and confidence < previous["confidence"]:
                continue

            if postprocess["corregida"]:
                print(
                    "[OCR] placa corregida "
                    f"{postprocess['texto_limpio']} -> {postprocess['texto_corregido']}"
                )

            self.plate_memory.store(key, {
                "text": corrected_text,
                "original_text": postprocess["texto_original"],
                "clean_text": postprocess["texto_limpio"],
                "confidence": confidence,
                "characters": [
                    {"value": char["value"], "confidence": char["confidence"]}
                    for char in characters
                ],
                "postprocess": postprocess,
                "seen_at": time.monotonic(),
            })

    def _apply_cached_plate_text(self, detection):
        cached = self.plate_memory.get(detection)
        if not cached:
            return detection

        detection["plate_text"] = cached["text"]
        detection["plate_original_text"] = cached.get("original_text", "")
        detection["plate_clean_text"] = cached.get("clean_text", "")
        detection["plate_text_confidence"] = cached["confidence"]
        detection["characters"] = cached.get("characters", [])
        detection["plate_postprocess"] = cached.get("postprocess")
        return detection

    def _ocr_candidate_ids(self, detections):
        if self.plate_reader is None or self.ocr_max_plates_per_frame <= 0:
            return set()

        candidates = [
            item
            for item in detections
            if item.get("confidence", 0.0) >= self.ocr_min_detection_confidence
            and item.get("ocr_zone_active", False)
        ]
        candidates.sort(
            key=lambda item: (
                0 if self.plate_memory.get(item) else 1,
                item.get("confidence", 0.0),
            ),
            reverse=True,
        )
        return {id(item) for item in candidates[: self.ocr_max_plates_per_frame]}

    def _should_run_ocr(self, detection):
        if self.plate_reader is None:
            return False

        if detection.get("_edge_clipped", False):
            return False

        if float(detection.get("_crop_cut_risk", 0.0)) >= 0.72:
            return False

        if float(detection.get("_crop_ghost_risk", 0.0)) >= 0.74:
            return False

        key = self.plate_memory.key_for(detection)
        if key in self.pending_ocr:
            return False

        if "_crop_quality" in detection and float(detection.get("_crop_quality", 0.0)) < 18.0:
            return False

        cached = self.plate_memory.get_raw(key)
        interval = (
            self.ocr_interval_seconds
            if cached and cached.get("text")
            else self.ocr_retry_interval_seconds
        )
        last_attempt = max(
            self.last_ocr_attempts.get(key, 0.0),
            self.last_ocr_attempt_at,
        )
        return time.monotonic() - last_attempt >= interval

    def _mark_ocr_attempt(self, detection):
        now = time.monotonic()
        self.last_ocr_attempt_at = now
        self.last_ocr_attempts[self.plate_memory.key_for(detection)] = now

    def _stabilize_plate_text(self, detection, text, confidence, characters):
        key = self.plate_memory.key_for(detection)
        previous = self.plate_memory.get_raw(key)
        postprocess = postprocess_ecuador_plate(text, confidence, characters)
        clean_text = postprocess["texto_corregido"]

        if clean_text and (
            previous is None
            or len(clean_text) > len(previous["text"])
            or confidence >= previous["confidence"]
        ):
            stored_characters = [
                {"value": char["value"], "confidence": char["confidence"]}
                for char in characters
            ]
            self.plate_memory.store(key, {
                "text": clean_text,
                "original_text": postprocess["texto_original"],
                "clean_text": postprocess["texto_limpio"],
                "confidence": confidence,
                "characters": stored_characters,
                "postprocess": postprocess,
                "seen_at": time.monotonic(),
            })
            detection["plate_original_text"] = postprocess["texto_original"]
            detection["plate_clean_text"] = postprocess["texto_limpio"]
            detection["plate_postprocess"] = postprocess
            return clean_text, confidence, stored_characters

        if previous is not None and time.monotonic() - previous["seen_at"] < 5:
            previous["seen_at"] = time.monotonic()
            return (
                previous["text"],
                previous["confidence"],
                previous.get("characters", []),
            )

        return clean_text, confidence, characters

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

    def _prepare_crop_items(self, crops, frame_bytes):
        prepared = []
        for item in crops:
            detection = item["detection"]
            evidence_frame, evidence_crop, evidence_detection = self.activity_tracker.get_evidence(detection.get("track_id"))
            crop_for_evidence = evidence_crop if evidence_crop is not None else item["crop"]
            crop_bytes = encode_jpeg(crop_for_evidence)
            if crop_bytes is None:
                continue

            incident_frame_bytes = self._incident_frame_bytes(
                evidence_detection or detection,
                frame_bytes,
                evidence_frame=evidence_frame,
            )
            fuzzy_result = self._evaluate_incident(detection, incident_frame_bytes, crop_bytes)
            if fuzzy_result is not None:
                detection["fuzzy_result"] = fuzzy_result

            prepared.append(
                {
                    "detection": detection,
                    "bytes": encode_jpeg(item["crop"]) or crop_bytes,
                }
            )
        return prepared

    def _incident_frame_bytes(self, detection, fallback_frame_bytes, evidence_frame=None):
        snapshot_frame = evidence_frame if evidence_frame is not None else detection.get("_snapshot_frame")
        if snapshot_frame is None:
            return fallback_frame_bytes

        try:
            speed_lines = self.speed_estimator.lines_for_frame(snapshot_frame.shape)
            evidence_frame, _, _ = process_frame(
                snapshot_frame,
                [detection],
                speed_lines=speed_lines,
                ocr_zone=self.ocr_zone_for_frame(snapshot_frame.shape),
                stats=None,
            )
            return encode_jpeg(evidence_frame, quality=82) or fallback_frame_bytes
        except Exception:
            return fallback_frame_bytes

    def _evaluate_incident(self, detection, frame_bytes, crop_bytes):
        if self.incident_service is None or frame_bytes is None:
            return None
        try:
            return self.incident_service.evaluate_detection(
                detection=detection,
                frame_bytes=frame_bytes,
                crop_bytes=crop_bytes,
                source_label=self.source_label,
            )
        except Exception:
            return None

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
            "plate_original_text": detection.get("plate_original_text") or "",
            "plate_clean_text": detection.get("plate_clean_text") or "",
            "plate_text_confidence": detection.get("plate_text_confidence", 0.0),
            "plate_postprocess": detection.get("plate_postprocess"),
            "characters": detection.get("characters", []),
            "ocr_zone_active": detection.get("ocr_zone_active", False),
            "speed_kmh": detection.get("speed_kmh"),
            "speed_status": detection.get("speed_status", "esperando cruce"),
            "fuzzy_result": detection.get("fuzzy_result"),
        }

    @staticmethod
    def _multipart_payload(frame_bytes):
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
