from pathlib import Path

from ultralytics import YOLO

from src.device import device_label, resolve_inference_device


class PlateDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.15,
        image_size=640,
        device="auto",
        tracker="bytetrack.yaml",
        iou_threshold=0.55,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = resolve_inference_device(device)
        self.device_label = device_label(self.device)
        self.tracker = tracker
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo configurado: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self._class_name(class_id),
                }
            )

        return self._deduplicate_detections(detections)

    def track(self, frame):
        result = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )[0]

        return self._deduplicate_detections(self._result_to_detections(result))

    def _result_to_detections(self, result):
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            track_id = None

            if confidence < self.confidence_threshold:
                continue

            if box.id is not None:
                track_id = int(box.id[0])

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self._class_name(class_id),
                    "track_id": track_id,
                }
            )

        return detections

    def _deduplicate_detections(self, detections):
        kept = []
        for detection in sorted(
            detections,
            key=lambda item: item.get("confidence", 0.0),
            reverse=True,
        ):
            duplicate_index = None
            for index, current in enumerate(kept):
                if self._same_plate_candidate(detection, current):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                kept.append(detection)
                continue

            current = kept[duplicate_index]
            if self._prefer_detection(detection, current):
                kept[duplicate_index] = detection

        return sorted(kept, key=lambda item: item.get("confidence", 0.0), reverse=True)

    def _same_plate_candidate(self, first, second):
        overlap = self._box_iou(first, second)
        if overlap >= self.iou_threshold:
            return True

        containment = self._intersection_over_smaller_box(first, second)
        if containment >= 0.72:
            return True

        first_center = self._center(first)
        second_center = self._center(second)
        first_width, first_height = self._size(first)
        second_width, second_height = self._size(second)
        distance = (
            (first_center[0] - second_center[0]) ** 2
            + (first_center[1] - second_center[1]) ** 2
        ) ** 0.5
        distance_limit = max(first_width, second_width, first_height, second_height) * 0.24
        width_ratio = min(first_width, second_width) / max(first_width, second_width, 1.0)
        height_ratio = min(first_height, second_height) / max(first_height, second_height, 1.0)

        return distance <= distance_limit and width_ratio >= 0.62 and height_ratio >= 0.62

    @staticmethod
    def _prefer_detection(candidate, current):
        confidence_gap = candidate.get("confidence", 0.0) - current.get("confidence", 0.0)
        if abs(confidence_gap) > 0.12:
            return confidence_gap > 0

        candidate_area = PlateDetector._area(candidate)
        current_area = PlateDetector._area(current)
        if candidate_area <= current_area * 0.80:
            return True
        if current_area <= candidate_area * 0.80:
            return False

        return confidence_gap > 0

    @staticmethod
    def _size(detection):
        return (
            max(1.0, float(detection["x2"]) - float(detection["x1"])),
            max(1.0, float(detection["y2"]) - float(detection["y1"])),
        )

    @staticmethod
    def _area(detection):
        width, height = PlateDetector._size(detection)
        return width * height

    @staticmethod
    def _center(detection):
        return (
            (float(detection["x1"]) + float(detection["x2"])) / 2,
            (float(detection["y1"]) + float(detection["y2"])) / 2,
        )

    @staticmethod
    def _box_iou(first, second):
        x1 = max(float(first["x1"]), float(second["x1"]))
        y1 = max(float(first["y1"]), float(second["y1"]))
        x2 = min(float(first["x2"]), float(second["x2"]))
        y2 = min(float(first["y2"]), float(second["y2"]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0

        return inter / (PlateDetector._area(first) + PlateDetector._area(second) - inter)

    @staticmethod
    def _intersection_over_smaller_box(first, second):
        x1 = max(float(first["x1"]), float(second["x1"]))
        y1 = max(float(first["y1"]), float(second["y1"]))
        x2 = min(float(first["x2"]), float(second["x2"]))
        y2 = min(float(first["y2"]), float(second["y2"]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0

        return inter / max(1.0, min(PlateDetector._area(first), PlateDetector._area(second)))

    def _class_name(self, class_id):
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).replace("_", " ")
        if 0 <= class_id < len(names):
            return str(names[class_id]).replace("_", " ")
        return str(class_id)
