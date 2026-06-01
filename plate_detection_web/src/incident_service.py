import json
import random
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.fuzzy_mamdani import evaluate_speed, save_fuzzy_artifacts


BRANDS = {
    "Toyota": ["Corolla", "Yaris", "Hilux", "RAV4"],
    "Chevrolet": ["Spark", "Sail", "Aveo", "Tracker"],
    "Kia": ["Rio", "Sportage", "Picanto", "Cerato"],
    "Hyundai": ["Tucson", "Accent", "Elantra", "Santa Fe"],
    "Nissan": ["Versa", "Sentra", "Kicks", "Frontier"],
}
COLORS = ["blanco", "gris", "negro", "rojo", "azul", "plata"]


class IncidentService:
    def __init__(self, db_path, static_dir, cooldown_seconds=45):
        self.db_path = db_path
        self.static_dir = static_dir
        self.incident_root = static_dir / "incidencias"
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.cooldowns = {}
        self._init_storage()

    def _init_storage(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.incident_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    plate_text TEXT NOT NULL,
                    characters_json TEXT NOT NULL,
                    detection_confidence REAL NOT NULL,
                    speed_kmh REAL NOT NULL,
                    risk_label TEXT NOT NULL,
                    penalty_hours REAL NOT NULL,
                    status TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    frame_path TEXT NOT NULL,
                    crop_path TEXT NOT NULL,
                    clip_path TEXT,
                    vehicle_brand TEXT NOT NULL,
                    vehicle_model TEXT NOT NULL,
                    vehicle_color TEXT NOT NULL,
                    fuzzy_json TEXT NOT NULL,
                    graph_input_path TEXT NOT NULL,
                    graph_output_path TEXT NOT NULL,
                    graph_aggregation_path TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def evaluate_detection(self, detection, frame_bytes, crop_bytes, source_label):
        speed = detection.get("speed_kmh")
        if speed is None:
            return None

        fuzzy = evaluate_speed(speed)
        public = {
            "label": fuzzy["label"],
            "penalizacion_horas": fuzzy["penalizacion_horas"],
            "is_safe": fuzzy["is_safe"],
            "rules": [
                rule for rule in fuzzy["rules"] if rule["active"]
            ],
            "input_memberships": fuzzy["input_memberships"],
        }

        if fuzzy["is_safe"]:
            return {
                **public,
                "penalizacion_horas": 0.0,
                "saved": False,
                "message": "Dentro del rango permitido",
            }

        key = self._cooldown_key(detection)
        if self._is_on_cooldown(key):
            return {**public, "saved": False, "message": "Incidencia reciente"}

        self.cooldowns[key] = time.monotonic()
        incident_id = str(uuid.uuid4())
        payload = self._build_payload(
            incident_id=incident_id,
            detection=detection,
            fuzzy=fuzzy,
            frame_bytes=frame_bytes,
            crop_bytes=crop_bytes,
            source_label=source_label,
        )
        self.executor.submit(self._save_incident, payload)
        return {
            **public,
            "saved": True,
            "incident_id": incident_id,
            "message": "Incidencia registrada",
        }

    def list_incidents(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, plate_text, speed_kmh, risk_label,
                       penalty_hours, detection_confidence, vehicle_brand,
                       vehicle_model, vehicle_color
                FROM incidents
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_incident(self, incident_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = ?",
                (incident_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["characters"] = json.loads(item["characters_json"])
        item["fuzzy"] = json.loads(item["fuzzy_json"])
        item["active_rules"] = [
            rule for rule in item["fuzzy"].get("rules", []) if rule.get("active")
        ]
        item["vehicle"] = {
            "brand": item["vehicle_brand"],
            "model": item["vehicle_model"],
            "color": item["vehicle_color"],
            "plate": item["plate_text"],
        }
        return item

    def _save_incident(self, payload):
        try:
            folder = self.incident_root / payload["id"]
            folder.mkdir(parents=True, exist_ok=True)
            frame_path = folder / "frame_principal.jpg"
            crop_path = folder / "recorte_placa.jpg"
            frame_path.write_bytes(payload["frame_bytes"])
            crop_path.write_bytes(payload["crop_bytes"])
            graphs = save_fuzzy_artifacts(payload["fuzzy"], folder)

            record = {
                **payload,
                "frame_path": self._static_rel(frame_path),
                "crop_path": self._static_rel(crop_path),
                "clip_path": "",
                "graph_input_path": self._static_rel(graphs["input_graph"]),
                "graph_output_path": self._static_rel(graphs["output_graph"]),
                "graph_aggregation_path": self._static_rel(graphs["aggregation_graph"]),
                "fuzzy_json": json.dumps(
                    {
                        **payload["fuzzy"],
                        "graph_paths": {
                            "input": self._static_rel(graphs["input_graph"]),
                            "output": self._static_rel(graphs["output_graph"]),
                            "aggregation": self._static_rel(graphs["aggregation_graph"]),
                        },
                    },
                    ensure_ascii=True,
                ),
            }
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO incidents (
                        id, created_at, plate_text, characters_json,
                        detection_confidence, speed_kmh, risk_label,
                        penalty_hours, status, source_label, frame_path,
                        crop_path, clip_path, vehicle_brand, vehicle_model,
                        vehicle_color, fuzzy_json, graph_input_path,
                        graph_output_path, graph_aggregation_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["id"],
                        record["created_at"],
                        record["plate_text"],
                        json.dumps(record["characters"], ensure_ascii=True),
                        record["detection_confidence"],
                        record["speed_kmh"],
                        record["risk_label"],
                        record["penalty_hours"],
                        record["status"],
                        record["source_label"],
                        record["frame_path"],
                        record["crop_path"],
                        record["clip_path"],
                        record["vehicle"]["brand"],
                        record["vehicle"]["model"],
                        record["vehicle"]["color"],
                        record["fuzzy_json"],
                        record["graph_input_path"],
                        record["graph_output_path"],
                        record["graph_aggregation_path"],
                    ),
                )
                conn.commit()
        except Exception:
            # El stream no debe caerse por un fallo de evidencia o persistencia.
            return

    def _build_payload(self, incident_id, detection, fuzzy, frame_bytes, crop_bytes, source_label):
        plate_text = detection.get("plate_text") or "No reconocida"
        return {
            "id": incident_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "plate_text": plate_text,
            "characters": detection.get("characters", []),
            "detection_confidence": float(detection.get("confidence", 0.0)),
            "speed_kmh": float(detection.get("speed_kmh") or 0.0),
            "risk_label": fuzzy["label"],
            "penalty_hours": float(fuzzy["penalizacion_horas"]),
            "status": fuzzy["label"].lower().replace(" ", "_"),
            "source_label": source_label or "",
            "frame_bytes": frame_bytes,
            "crop_bytes": crop_bytes,
            "vehicle": self._fake_vehicle(plate_text),
            "fuzzy": fuzzy,
        }

    def _fake_vehicle(self, plate_text):
        seed = sum(ord(char) for char in plate_text)
        rng = random.Random(seed)
        brand = rng.choice(list(BRANDS.keys()))
        return {
            "brand": brand,
            "model": rng.choice(BRANDS[brand]),
            "color": rng.choice(COLORS),
            "plate": plate_text,
        }

    def _cooldown_key(self, detection):
        plate = detection.get("plate_text")
        if plate:
            return f"plate:{plate}"
        track_id = detection.get("track_id")
        if track_id is not None:
            return f"track:{track_id}"
        return f"pos:{round(detection.get('x1', 0) / 80)}:{round(detection.get('y1', 0) / 80)}"

    def _is_on_cooldown(self, key):
        now = time.monotonic()
        expired = [
            item_key
            for item_key, seen_at in self.cooldowns.items()
            if now - seen_at > self.cooldown_seconds
        ]
        for item_key in expired:
            self.cooldowns.pop(item_key, None)
        return key in self.cooldowns

    def _static_rel(self, path):
        return str(path.relative_to(self.static_dir)).replace("\\", "/")

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
