import base64
from threading import Lock, Thread
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import Config
from src.detector import PlateDetector
from src.fuzzy_mamdani import get_mamdani_config, set_mamdani_config
from src.incident_service import IncidentService
from src.plate_reader import PlateReader
from src.video_stream import VideoStream


app = Flask(__name__)
Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
incident_service = IncidentService(
    db_path=Config.INCIDENT_DB_PATH,
    static_dir=Config.STATIC_DIR,
    cooldown_seconds=Config.INCIDENT_COOLDOWN_SECONDS,
    email_config={
        "enabled": Config.EMAIL_INCIDENTS_ENABLED,
        "recipient": Config.INCIDENT_EMAIL_TO,
        "smtp_host": Config.SMTP_HOST,
        "smtp_port": Config.SMTP_PORT,
        "smtp_user": Config.SMTP_USER,
        "smtp_password": Config.SMTP_PASSWORD,
        "smtp_from": Config.SMTP_FROM,
    },
)

detector_error = None
reader_error = None
try:
    detector = PlateDetector(
        model_path=Config.MODEL_PATH,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        image_size=Config.IMAGE_SIZE,
        device=Config.DEVICE,
        tracker=Config.TRACKER,
        iou_threshold=Config.IOU_THRESHOLD,
    )
except Exception as exc:
    detector = None
    detector_error = str(exc)

try:
    plate_reader = PlateReader(
        model_path=Config.CHARACTER_MODEL_PATH,
        confidence_threshold=Config.CHARACTER_CONFIDENCE_THRESHOLD,
        image_size=Config.CHARACTER_IMAGE_SIZE,
        device=Config.DEVICE,
        max_variants=Config.OCR_VARIANTS,
    )
except Exception as exc:
    plate_reader = None
    reader_error = str(exc)

stream = VideoStream(
    detector=detector,
    plate_reader=plate_reader,
    source=Config.VIDEO_SOURCE,
    speed_line_a_y=Config.SPEED_LINE_A_Y,
    speed_line_b_y=Config.SPEED_LINE_B_Y,
    speed_line_a_right_y=Config.SPEED_LINE_A_RIGHT_Y,
    speed_line_b_right_y=Config.SPEED_LINE_B_RIGHT_Y,
    speed_roi_x1=Config.SPEED_ROI_X1,
    speed_roi_x2=Config.SPEED_ROI_X2,
    speed_distance_meters=Config.SPEED_DISTANCE_METERS,
    speed_direction=Config.SPEED_DIRECTION,
    speed_line_hysteresis_px=Config.SPEED_LINE_HYSTERESIS_PX,
    speed_min_travel_time=Config.SPEED_MIN_TRAVEL_TIME,
    speed_max_travel_time=Config.SPEED_MAX_TRAVEL_TIME,
    speed_min_movement_px=Config.SPEED_MIN_MOVEMENT_PX,
    speed_min_partial_progress=Config.SPEED_MIN_PARTIAL_PROGRESS,
    target_fps=Config.STREAM_TARGET_FPS,
    stream_max_width=Config.STREAM_MAX_WIDTH,
    detection_every_n_frames=Config.DETECTION_EVERY_N_FRAMES,
    live_detection_interval_seconds=Config.LIVE_DETECTION_INTERVAL_SECONDS,
    live_detection_mode=Config.LIVE_DETECTION_MODE,
    ocr_interval_seconds=Config.OCR_INTERVAL_SECONDS,
    ocr_retry_interval_seconds=Config.OCR_RETRY_INTERVAL_SECONDS,
    ocr_max_plates_per_frame=Config.OCR_MAX_PLATES_PER_FRAME,
    ocr_min_detection_confidence=Config.OCR_MIN_DETECTION_CONFIDENCE,
    camera_backend=Config.CAMERA_BACKEND,
    camera_width=Config.CAMERA_WIDTH,
    camera_height=Config.CAMERA_HEIGHT,
    camera_fps=Config.CAMERA_FPS,
    camera_fourcc=Config.CAMERA_FOURCC,
    incident_service=incident_service,
    detector_error=detector_error or reader_error,
)


camera_sources_cache = None
camera_sources_scan_running = False
camera_sources_lock = Lock()


SPEED_CONFIG_ENV_KEYS = {
    "line_a_left_y": "SPEED_LINE_A_LEFT_Y",
    "line_a_right_y": "SPEED_LINE_A_RIGHT_Y",
    "line_b_left_y": "SPEED_LINE_B_LEFT_Y",
    "line_b_right_y": "SPEED_LINE_B_RIGHT_Y",
    "roi_x1": "SPEED_ROI_X1",
    "roi_x2": "SPEED_ROI_X2",
    "distance_meters": "SPEED_DISTANCE_METERS",
}
EMAIL_CONFIG_ENV_KEYS = {
    "enabled": "EMAIL_INCIDENTS_ENABLED",
}


def camera_backend_candidates():
    backends = {
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
        "auto": cv2.CAP_ANY,
    }
    preferred = backends.get(str(Config.CAMERA_BACKEND).strip().lower(), cv2.CAP_MSMF)
    ordered = [preferred, cv2.CAP_MSMF, cv2.CAP_ANY]
    unique = []
    for backend in ordered:
        if backend not in unique:
            unique.append(backend)
    return unique


def fallback_camera_sources():
    configured_source = Config.VIDEO_SOURCE if isinstance(Config.VIDEO_SOURCE, int) else 0
    values = list(range(max(2, Config.CAMERA_SCAN_LIMIT)))
    if configured_source not in values:
        values.insert(0, configured_source)
    return [{"type": "camera", "value": value, "label": f"Camara {value}"} for value in values]


def scan_camera_sources():
    global camera_sources_cache, camera_sources_scan_running

    cameras = fallback_camera_sources()
    try:
        discovered = []
        backends = camera_backend_candidates()

        for index in range(Config.CAMERA_SCAN_LIMIT):
            opened = False
            for backend in backends:
                capture = cv2.VideoCapture(index, backend)
                opened = capture.isOpened()
                capture.release()
                if opened:
                    break

            if opened:
                discovered.append({"type": "camera", "value": index, "label": f"Camara {index}"})

        if discovered:
            known_values = {item["value"] for item in discovered}
            for fallback in fallback_camera_sources():
                if fallback["value"] not in known_values:
                    discovered.append(fallback)
            cameras = sorted(discovered, key=lambda item: int(item["value"]))
    finally:
        with camera_sources_lock:
            camera_sources_cache = cameras
            camera_sources_scan_running = False


def start_camera_scan_if_needed():
    global camera_sources_scan_running

    with camera_sources_lock:
        if camera_sources_scan_running:
            return
        camera_sources_scan_running = True

    Thread(target=scan_camera_sources, daemon=True).start()


def update_env_values(values):
    env_path = Path(__file__).resolve().parent / ".env"
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    pending = {key: str(value) for key, value in values.items()}
    updated_lines = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue

        key, _ = line.split("=", 1)
        key = key.strip()
        if key in pending:
            updated_lines.append(f"{key}={pending.pop(key)}")
        else:
            updated_lines.append(line)

    if pending:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        for key, value in pending.items():
            updated_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def jpeg_data_url(image, quality=82):
    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def available_camera_sources():
    with camera_sources_lock:
        cached = [dict(item) for item in camera_sources_cache] if camera_sources_cache else None

    if cached is not None:
        return cached

    start_camera_scan_if_needed()
    return fallback_camera_sources()


@app.route("/")
def index():
    return render_template(
        "index.html",
        detector_error=detector_error,
        reader_error=reader_error,
        video_aspect_ratio=Config.VIDEO_ASPECT_RATIO,
        video_max_width=Config.VIDEO_MAX_WIDTH,
        active_page="foto_radar",
    )


@app.route("/incidencias")
def incidencias():
    return render_template(
        "incidencias.html",
        incidents=incident_service.list_incidents(),
        active_page="incidencias",
    )


@app.route("/incidencias/<incident_id>")
def incidencia_detalle(incident_id):
    incident = incident_service.get_incident(incident_id)
    if incident is None:
        return render_template("incidencia_no_encontrada.html", active_page="incidencias"), 404
    return render_template(
        "incidencia_detalle.html",
        incident=incident,
        active_page="incidencias",
        email_status=request.args.get("email_status", ""),
    )


@app.route("/incidencias/<incident_id>/enviar_correo", methods=["POST"])
def enviar_correo_incidencia(incident_id):
    result = incident_service.send_incident_email(incident_id)
    if result.get("already_sent"):
        status = "already"
    elif result.get("ok"):
        status = "sent"
    else:
        status = "error"

    return redirect(url_for("incidencia_detalle", incident_id=incident_id, email_status=status))


@app.route("/configuracion")
def configuracion():
    return render_template(
        "configuracion.html",
        active_page="configuracion",
        video_aspect_ratio=Config.VIDEO_ASPECT_RATIO,
        video_max_width=Config.VIDEO_MAX_WIDTH,
    )


@app.route("/ocr_detalle/<int:index>")
def ocr_detalle(index):
    status = stream.current_status()
    detections = status.get("detections", [])
    detection = detections[index] if 0 <= index < len(detections) else None
    crop_bytes = stream.current_crop_at(index)
    crop = cv2.imdecode(np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    steps = []
    if plate_reader is not None and crop is not None:
        for step in plate_reader.preprocessing_debug(crop):
            steps.append(
                {
                    **step,
                    "image_url": jpeg_data_url(step["image"]),
                    "image": None,
                }
            )

    return render_template(
        "ocr_detalle.html",
        active_page="foto_radar",
        index=index,
        detection=detection,
        steps=steps,
        reader_error=reader_error,
        ocr_variants=Config.OCR_VARIANTS,
        character_image_size=Config.CHARACTER_IMAGE_SIZE,
    )


@app.route("/api/speed_config", methods=["GET", "POST"])
def speed_config():
    if request.method == "GET":
        return jsonify({"ok": True, "config": stream.current_speed_config()})

    payload = request.get_json(silent=True) or {}
    accepted = {}
    numeric_fields = {
        "line_a_left_y",
        "line_a_right_y",
        "line_b_left_y",
        "line_b_right_y",
        "roi_x1",
        "roi_x2",
        "distance_meters",
    }
    for field in numeric_fields:
        if field not in payload:
            continue
        try:
            accepted[field] = float(payload[field])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": f"Valor invalido: {field}"}), 400

    updated = stream.update_speed_config(**accepted)
    env_values = {
        env_key: f"{updated[field]:.4f}".rstrip("0").rstrip(".")
        for field, env_key in SPEED_CONFIG_ENV_KEYS.items()
        if field in accepted
    }
    if env_values:
        update_env_values(env_values)

    return jsonify({"ok": True, "config": updated})


@app.route("/api/mamdani_config", methods=["GET", "POST"])
def mamdani_config():
    if request.method == "GET":
        return jsonify({"ok": True, "config": get_mamdani_config()})

    payload = request.get_json(silent=True) or {}
    try:
        updated = set_mamdani_config(payload)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "config": updated})


@app.route("/api/email_config", methods=["GET", "POST"])
def email_config():
    if request.method == "GET":
        current = incident_service.current_email_config()
        return jsonify({"ok": True, "config": {"enabled": current["enabled"]}})

    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))
    updated = incident_service.update_email_config(enabled=enabled)
    update_env_values({"EMAIL_INCIDENTS_ENABLED": "1" if updated["enabled"] else "0"})
    return jsonify({"ok": True, "config": {"enabled": updated["enabled"]}})


@app.route("/video_feed")
def video_feed():
    return Response(
        stream.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/plate_crop")
def plate_crop():
    return Response(stream.current_crop(), mimetype="image/jpeg")


@app.route("/plate_crop/<int:index>")
def plate_crop_at(index):
    return Response(stream.current_crop_at(index), mimetype="image/jpeg")


@app.route("/current_frame")
def current_frame():
    return Response(stream.current_frame(), mimetype="image/jpeg")


@app.route("/status")
def status():
    return jsonify(stream.current_status())


@app.route("/sources")
def sources():
    return jsonify(
        {
            "ok": True,
            "sources": available_camera_sources()
            + [{"type": "video", "value": "video", "label": "Video"}],
        }
    )


@app.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    paused = stream.toggle_pause()
    return jsonify({"ok": True, "paused": paused})


@app.route("/play", methods=["POST"])
def play():
    paused = stream.set_paused(False)
    return jsonify({"ok": True, "paused": paused})


@app.route("/pause", methods=["POST"])
def pause():
    paused = stream.set_paused(True)
    return jsonify({"ok": True, "paused": paused})


@app.route("/seek", methods=["POST"])
def seek():
    payload = request.get_json(silent=True) or {}
    if "seconds" in payload:
        position = stream.seek_to(payload["seconds"])
    else:
        position = stream.seek_relative(payload.get("delta", 0))
    return jsonify({"ok": True, "position_seconds": position})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No se recibio ningun video."}), 400

    extension = Path(file.filename).suffix.lower()
    if extension not in Config.ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"ok": False, "error": "Formato de video no soportado."}), 400

    filename = secure_filename(file.filename)
    destination = Config.UPLOAD_DIR / filename
    counter = 1
    while destination.exists():
        destination = Config.UPLOAD_DIR / f"{destination.stem}_{counter}{extension}"
        counter += 1

    file.save(destination)
    stream.set_source(str(destination), source_label=destination.name)
    return jsonify(
        {
            "ok": True,
            "filename": destination.name,
            "video_url": url_for("static", filename=f"captures/{destination.name}"),
        }
    )


@app.route("/use_camera", methods=["POST"])
def use_camera():
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", Config.VIDEO_SOURCE)

    try:
        source = int(source)
    except (TypeError, ValueError):
        source = Config.VIDEO_SOURCE

    stream.set_source(source, source_label=f"Camara {source}")
    return jsonify({"ok": True, "source": source, "label": f"Camara {source}"})


if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False,
    )
