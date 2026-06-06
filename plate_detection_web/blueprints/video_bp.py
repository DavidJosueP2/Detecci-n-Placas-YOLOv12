from pathlib import Path
from threading import Lock, Thread

import cv2
from flask import Blueprint, Response, current_app, jsonify, request, url_for
from werkzeug.utils import secure_filename

from config import Config

video_bp = Blueprint("video", __name__)

camera_sources_cache = None
camera_sources_scan_running = False
camera_sources_lock = Lock()


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


def available_camera_sources():
    with camera_sources_lock:
        cached = [dict(item) for item in camera_sources_cache] if camera_sources_cache else None

    if cached is not None:
        return cached

    start_camera_scan_if_needed()
    return fallback_camera_sources()


@video_bp.route("/video_feed")
def video_feed():
    return Response(
        current_app.stream.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@video_bp.route("/plate_crop/<int:index>")
def plate_crop_at(index):
    return Response(current_app.stream.current_crop_at(index), mimetype="image/jpeg")


@video_bp.route("/status")
def status():
    return jsonify(current_app.stream.current_status())


@video_bp.route("/sources")
def sources():
    return jsonify(
        {
            "ok": True,
            "sources": available_camera_sources()
            + [{"type": "video", "value": "video", "label": "Video"}],
        }
    )


@video_bp.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    paused = current_app.stream.toggle_pause()
    return jsonify({"ok": True, "paused": paused})


@video_bp.route("/play", methods=["POST"])
def play():
    paused = current_app.stream.set_paused(False)
    return jsonify({"ok": True, "paused": paused})


@video_bp.route("/pause", methods=["POST"])
def pause():
    paused = current_app.stream.set_paused(True)
    return jsonify({"ok": True, "paused": paused})


@video_bp.route("/seek", methods=["POST"])
def seek():
    payload = request.get_json(silent=True) or {}
    if "seconds" in payload:
        position = current_app.stream.seek_to(payload["seconds"])
    else:
        position = current_app.stream.seek_relative(payload.get("delta", 0))
    return jsonify({"ok": True, "position_seconds": position})


@video_bp.route("/upload_video", methods=["POST"])
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
    current_app.stream.set_source(str(destination), source_label=destination.name)
    return jsonify(
        {
            "ok": True,
            "filename": destination.name,
            "video_url": url_for("static", filename=f"captures/{destination.name}"),
        }
    )


@video_bp.route("/use_camera", methods=["POST"])
def use_camera():
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", Config.VIDEO_SOURCE)

    try:
        source = int(source)
    except (TypeError, ValueError):
        source = Config.VIDEO_SOURCE

    current_app.stream.set_source(source, source_label=f"Camara {source}")
    return jsonify({"ok": True, "source": source, "label": f"Camara {source}"})
