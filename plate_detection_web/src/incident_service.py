import json
import random
import smtplib
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage
from threading import Lock

from src.fuzzy_mamdani import evaluate_speed, save_fuzzy_artifacts


BRANDS = {
    "Toyota": ["Corolla", "Yaris", "Hilux", "RAV4"],
    "Chevrolet": ["Spark", "Sail", "Aveo", "Tracker"],
    "Kia": ["Rio", "Sportage", "Picanto", "Cerato"],
    "Hyundai": ["Tucson", "Accent", "Elantra", "Santa Fe"],
    "Nissan": ["Versa", "Sentra", "Kicks", "Frontier"],
}
COLORS = ["blanco", "gris", "negro", "rojo", "azul", "plata"]
SMTP_PASSWORD_PLACEHOLDER = "TU_CONTRASEÑA_DE_APLICACION"


class IncidentService:
    def __init__(self, db_path, static_dir, cooldown_seconds=45, email_config=None):
        self.db_path = db_path
        self.static_dir = static_dir
        self.incident_root = static_dir / "incidencias"
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.cooldowns = {}
        self.email_lock = Lock()
        self.email_config = self._normalize_email_config(email_config or {})
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
                    graph_aggregation_path TEXT NOT NULL,
                    email_sent INTEGER NOT NULL DEFAULT 0,
                    email_sent_at TEXT
                )
                """
            )
            self._ensure_column(conn, "email_sent", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "email_sent_at", "TEXT")
            conn.commit()

    @staticmethod
    def _ensure_column(conn, column_name, definition):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(incidents)").fetchall()
        }
        if column_name not in columns:
            conn.execute(f"ALTER TABLE incidents ADD COLUMN {column_name} {definition}")

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
            message = (
                "Felicitacion normal"
                if fuzzy.get("bypass_mamdani")
                else "Dentro del rango permitido"
            )
            return {
                **public,
                "penalizacion_horas": 0.0,
                "saved": False,
                "message": message,
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
        item["email_sent"] = bool(item.get("email_sent"))
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
            if self._send_incident_email(record, frame_path, crop_path):
                self._mark_email_sent(record["id"])
        except Exception:
            # El stream no debe caerse por un fallo de evidencia o persistencia.
            return

    def send_incident_email(self, incident_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = ?",
                (incident_id,),
            ).fetchone()

        if row is None:
            return {"ok": False, "error": "Incidencia no encontrada"}

        record = dict(row)
        if record.get("email_sent"):
            return {"ok": True, "already_sent": True}

        frame_path = self.static_dir / record["frame_path"]
        crop_path = self.static_dir / record["crop_path"]
        if not self._send_incident_email(record, frame_path, crop_path, force=True):
            return {"ok": False, "error": "No se pudo enviar el correo"}

        self._mark_email_sent(incident_id)
        return {"ok": True, "already_sent": False}

    def _mark_email_sent(self, incident_id):
        sent_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "UPDATE incidents SET email_sent = 1, email_sent_at = ? WHERE id = ?",
                (sent_at, incident_id),
            )
            conn.commit()

    def current_email_config(self):
        with self.email_lock:
            config = dict(self.email_config)
        password = config.pop("smtp_password", "")
        config["smtp_password_set"] = bool(password and password != SMTP_PASSWORD_PLACEHOLDER)
        return config

    def update_email_config(self, **values):
        with self.email_lock:
            merged = {**self.email_config, **values}
            if not values.get("smtp_password"):
                merged["smtp_password"] = self.email_config.get("smtp_password", "")
            self.email_config = self._normalize_email_config(merged)
            config = dict(self.email_config)
        password = config.pop("smtp_password", "")
        config["smtp_password_set"] = bool(password and password != SMTP_PASSWORD_PLACEHOLDER)
        return config

    def _send_incident_email(self, record, frame_path, crop_path, force=False):
        with self.email_lock:
            config = dict(self.email_config)

        if not self._email_config_ready(config, require_enabled=not force):
            return

        message = self._build_email_message(record, config)
        self._attach_file(message, frame_path, "frame_principal.jpg")
        self._attach_file(message, crop_path, "recorte_placa.jpg")

        try:
            with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=18) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(config["smtp_user"], config["smtp_password"])
                smtp.send_message(message)
            return True
        except Exception:
            return False

    def _build_email_message(self, record, config):
        subject_plate = record.get("plate_text") or "No reconocida"
        message = EmailMessage()
        message["Subject"] = f"Grupo K | Incidencia foto radar | Placa {subject_plate}"
        message["From"] = config["smtp_from"]
        message["To"] = config["recipient"]

        text_body = (
            "Regularizacion vehicular - Notificacion de incidencia\n\n"
            "Grupo K\n"
            "Responsables: David Barragán y Ariel Paredes\n\n"
            f"Fecha/hora: {record['created_at']}\n"
            f"Placa: {record['plate_text']}\n"
            f"Velocidad registrada: {record['speed_kmh']:.1f} km/h\n"
            f"Nivel de riesgo: {record['risk_label']}\n"
            f"Penalizacion estimada: {record['penalty_hours']:.2f} h\n"
            f"Confianza de deteccion: {record['detection_confidence'] * 100:.1f}%\n"
            f"Fuente: {record['source_label']}\n\n"
            "Se adjuntan la imagen principal de la incidencia y el recorte de la placa."
        )
        message.set_content(text_body)
        message.add_alternative(self._email_html_body(record), subtype="html")
        return message

    @staticmethod
    def _email_html_body(record):
        confidence = record["detection_confidence"] * 100
        return f"""\
<!doctype html>
<html lang="es">
  <body style="margin:0;background:#f4f7fb;padding:0;font-family:Segoe UI,Arial,sans-serif;color:#0f172a;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f7fb;padding:28px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="680" cellspacing="0" cellpadding="0" style="width:680px;max-width:94%;background:#ffffff;border:1px solid #dbe3ee;">
            <tr>
              <td style="background:#084f99;padding:24px 28px;color:#ffffff;">
                <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:700;">Grupo K</div>
                <div style="font-size:24px;font-weight:700;margin-top:6px;">Regularizacion vehicular</div>
                <div style="font-size:14px;margin-top:8px;color:#d8ebff;">Notificacion formal de incidencia registrada</div>
              </td>
            </tr>
            <tr>
              <td style="padding:26px 28px 12px 28px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="width:50%;padding:0 10px 14px 0;">
                      <div style="font-size:12px;color:#64748b;text-transform:uppercase;font-weight:700;">Placa</div>
                      <div style="font-size:28px;font-weight:800;margin-top:4px;">{record['plate_text']}</div>
                    </td>
                    <td style="width:50%;padding:0 0 14px 10px;text-align:right;">
                      <div style="font-size:12px;color:#64748b;text-transform:uppercase;font-weight:700;">Nivel de riesgo</div>
                      <div style="display:inline-block;margin-top:6px;background:#fee2e2;color:#b91c1c;border:1px solid #fecaca;padding:8px 12px;font-size:16px;font-weight:800;">{record['risk_label']}</div>
                    </td>
                  </tr>
                </table>
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-top:8px;">
                  <tr>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;color:#475569;">Fecha/hora</td>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;text-align:right;font-weight:700;">{record['created_at']}</td>
                  </tr>
                  <tr>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;color:#475569;">Velocidad registrada</td>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;text-align:right;font-weight:700;">{record['speed_kmh']:.1f} km/h</td>
                  </tr>
                  <tr>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;color:#475569;">Penalizacion estimada</td>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;text-align:right;font-weight:700;">{record['penalty_hours']:.2f} h</td>
                  </tr>
                  <tr>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;color:#475569;">Confianza de deteccion</td>
                    <td style="border-top:1px solid #e2e8f0;padding:12px 0;text-align:right;font-weight:700;">{confidence:.1f}%</td>
                  </tr>
                  <tr>
                    <td style="border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;padding:12px 0;color:#475569;">Fuente</td>
                    <td style="border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;padding:12px 0;text-align:right;font-weight:700;">{record['source_label']}</td>
                  </tr>
                </table>
                <div style="margin-top:22px;padding:16px;background:#f8fafc;border:1px solid #e2e8f0;color:#334155;font-size:14px;line-height:1.55;">
                  Se adjuntan la evidencia principal de la incidencia y el recorte de la placa detectada para su revision.
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:18px 28px 26px 28px;color:#475569;font-size:13px;line-height:1.55;">
                <strong>Responsables:</strong> David Barragán y Ariel Paredes
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    @staticmethod
    def _attach_file(message, path, filename):
        if not path.exists():
            return
        message.add_attachment(
            path.read_bytes(),
            maintype="image",
            subtype="jpeg",
            filename=filename,
        )

    @staticmethod
    def _normalize_email_config(config):
        return {
            "enabled": bool(config.get("enabled", False)),
            "recipient": str(config.get("recipient", "") or "").strip(),
            "smtp_host": str(config.get("smtp_host", "smtp.office365.com") or "").strip(),
            "smtp_port": int(config.get("smtp_port", 587) or 587),
            "smtp_user": str(config.get("smtp_user", "") or "").strip(),
            "smtp_password": str(config.get("smtp_password", "") or ""),
            "smtp_from": str(config.get("smtp_from", config.get("smtp_user", "")) or "").strip(),
        }

    @staticmethod
    def _email_config_ready(config, require_enabled=True):
        checks = [
            config.get("recipient"),
            config.get("smtp_host"),
            config.get("smtp_port"),
            config.get("smtp_user"),
            config.get("smtp_password"),
            config.get("smtp_password") != SMTP_PASSWORD_PLACEHOLDER,
            config.get("smtp_from"),
        ]
        if require_enabled:
            checks.append(config.get("enabled"))
        return all(checks)

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
