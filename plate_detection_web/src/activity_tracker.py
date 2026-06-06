import time
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.frame_processor import encode_jpeg, process_frame


class ActivityTracker:
    def __init__(
        self,
        incident_service=None,
        speed_estimator=None,
        ocr_zone_fn=None,
        plate_reader=None,
        debug_root=None,
    ):
        self._tracks = {}
        self._incident_service = incident_service
        self._speed_estimator = speed_estimator
        self._ocr_zone_fn = ocr_zone_fn
        self._plate_reader = plate_reader
        self._debug_root = Path(debug_root) if debug_root is not None else None

    def update(self, detection, frame, crop, timestamp, source_label):
        track_id = detection.get("track_id")
        if track_id is None or crop is None or crop.size == 0:
            return

        key = f"track:{track_id}"
        now = time.monotonic()
        activity = self._tracks.get(key)
        if activity is None:
            activity = {
                "track_id": track_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "started_monotonic": now,
                "last_seen_monotonic": now,
                "last_timestamp": timestamp,
                "source_label": source_label,
                "plate_text": "",
                "plate_original_text": "",
                "characters": [],
                "plate_confidence": 0.0,
                "detection_confidence": 0.0,
                "speed_kmh": None,
                "best_score": -1.0,
                "best_quality": 0.0,
                "best_frame": None,
                "best_crop": None,
                "best_detection": None,
            }
            self._tracks[key] = activity

        activity["last_seen_monotonic"] = now
        activity["last_timestamp"] = timestamp
        activity["source_label"] = source_label
        if detection.get("plate_text"):
            activity["plate_text"] = detection.get("plate_text", "")
            activity["plate_original_text"] = detection.get("plate_original_text", "")
            activity["plate_confidence"] = float(detection.get("plate_text_confidence", 0.0))
            activity["characters"] = detection.get("characters", [])
        if detection.get("speed_kmh") is not None:
            activity["speed_kmh"] = float(detection["speed_kmh"])
        activity["detection_confidence"] = max(
            activity.get("detection_confidence", 0.0),
            float(detection.get("confidence", 0.0)),
        )

        edge_clipped = bool(detection.get("_edge_clipped", False))
        cut_risk = float(detection.get("_crop_cut_risk", 0.0))
        ghost_risk = float(detection.get("_crop_ghost_risk", 0.0))
        if edge_clipped or cut_risk >= 0.72:
            return

        width = max(1.0, float(detection.get("x2", 0.0)) - float(detection.get("x1", 0.0)))
        quality = float(detection.get("_crop_quality", 0.0))
        zone_bonus = 1000.0 if detection.get("ocr_zone_active") else 0.0
        ocr_bonus = 1800.0 if detection.get("plate_text") else 0.0
        ghost_penalty = max(0.0, ghost_risk - 0.48) * 2200.0
        score = (
            zone_bonus
            + ocr_bonus
            + min(width, 320.0) * 1.5
            + quality * 7.0
            + float(detection.get("confidence", 0.0)) * 80.0
            - ghost_penalty
        )
        if score > activity.get("best_score", -1.0):
            activity["best_score"] = score
            activity["best_quality"] = quality
            activity["best_frame"] = frame.copy()
            activity["best_crop"] = crop.copy()
            activity["best_detection"] = _evidence_detection(detection)

    def prune(self, max_age_seconds=4.0):
        now = time.monotonic()
        expired = [
            key
            for key, activity in self._tracks.items()
            if now - activity.get("last_seen_monotonic", now) > max_age_seconds
        ]
        for key in expired:
            self._finalize(key)

    def flush(self):
        for key in list(self._tracks):
            self._finalize(key)

    def get_evidence(self, track_id):
        if track_id is None:
            return None, None, None
        activity = self._tracks.get(f"track:{track_id}")
        if not activity:
            return None, None, None
        return (
            activity.get("best_frame"),
            activity.get("best_crop"),
            activity.get("best_detection"),
        )

    def _finalize(self, key):
        activity = self._tracks.pop(key, None)
        if not activity or self._incident_service is None:
            return
        if not hasattr(self._incident_service, "record_activity"):
            return
        if not str(activity.get("plate_text") or "").strip():
            return
        if activity.get("best_frame") is None or activity.get("best_crop") is None:
            return

        frame_bytes = self._frame_bytes(activity)
        crop_bytes = encode_jpeg(activity["best_crop"], quality=82)
        if frame_bytes is None or crop_bytes is None:
            return

        activity_id = str(uuid.uuid4())
        ended_at = datetime.now().isoformat(timespec="seconds")
        duration = max(
            0.0,
            float(activity.get("last_seen_monotonic", time.monotonic()))
            - float(activity.get("started_monotonic", time.monotonic())),
        )
        self._incident_service.record_activity(
            {
                "id": activity_id,
                "track_id": activity.get("track_id"),
                "source_label": activity.get("source_label", ""),
                "started_at": activity.get("started_at", ended_at),
                "ended_at": ended_at,
                "duration_seconds": duration,
                "plate_text": activity.get("plate_text", ""),
                "plate_original_text": activity.get("plate_original_text", ""),
                "characters": activity.get("characters", []),
                "plate_confidence": activity.get("plate_confidence", 0.0),
                "detection_confidence": activity.get("detection_confidence", 0.0),
                "speed_kmh": activity.get("speed_kmh"),
                "frame_bytes": frame_bytes,
                "crop_bytes": crop_bytes,
            }
        )

        if (
            self._plate_reader is not None
            and self._debug_root is not None
            and hasattr(self._plate_reader, "read_debug")
        ):
            best_crop = activity["best_crop"].copy()
            t = threading.Thread(
                target=self._save_debug_images,
                args=(activity_id, best_crop),
                daemon=True,
            )
            t.start()

    def _save_debug_images(self, activity_id: str, crop: np.ndarray):
        try:
            debug_dir = self._debug_root / activity_id / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # seg/ subfolder receives all intermediate segmentation stage images
            # (gray, CLAHE, denoised, binary variants, bboxes, chars strip, mosaic)
            debug = self._plate_reader.read_debug(crop, debug_dir=debug_dir / "seg")

            cv2.imwrite(str(debug_dir / "char_bboxes.jpg"), debug["bbox_image"])

            for i, img in enumerate(debug["char_images"]):
                upscaled = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(debug_dir / f"char_{i}.png"), upscaled)
        except Exception:
            pass

    def _frame_bytes(self, activity):
        frame = activity.get("best_frame")
        detection = activity.get("best_detection")
        if frame is None:
            return None

        if detection is None:
            return encode_jpeg(frame, quality=82)

        try:
            speed_lines = (
                self._speed_estimator.lines_for_frame(frame.shape)
                if self._speed_estimator is not None
                else []
            )
            ocr_zone = (
                self._ocr_zone_fn(frame.shape)
                if self._ocr_zone_fn is not None
                else None
            )
            evidence_frame, _, _ = process_frame(
                frame,
                [detection],
                speed_lines=speed_lines,
                ocr_zone=ocr_zone,
                stats=None,
            )
            return encode_jpeg(evidence_frame, quality=82)
        except Exception:
            return encode_jpeg(frame, quality=82)


def _evidence_detection(detection):
    return {key: value for key, value in detection.items() if not key.startswith("_")}
