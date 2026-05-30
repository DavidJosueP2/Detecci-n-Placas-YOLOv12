from pathlib import Path

from ultralytics import YOLO


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
        self.device = None if device == "auto" else device
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

        return detections

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

        return self._result_to_detections(result)

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

    def _class_name(self, class_id):
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).replace("_", " ")
        if 0 <= class_id < len(names):
            return str(names[class_id]).replace("_", " ")
        return str(class_id)
