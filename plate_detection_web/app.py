from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import Config
from src.detector import PlateDetector
from src.plate_reader import PlateReader
from src.video_stream import VideoStream


app = Flask(__name__)
Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    )
except Exception as exc:
    plate_reader = None
    reader_error = str(exc)

stream = VideoStream(
    detector=detector,
    plate_reader=plate_reader,
    source=Config.VIDEO_SOURCE,
    pixels_per_meter=Config.SPEED_PIXELS_PER_METER,
    speed_smoothing=Config.SPEED_SMOOTHING,
    detector_error=detector_error or reader_error,
)


def available_camera_sources():
    configured_source = Config.VIDEO_SOURCE if isinstance(Config.VIDEO_SOURCE, int) else 0
    cameras = []

    for index in range(Config.CAMERA_SCAN_LIMIT):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = capture.isOpened()
        capture.release()
        if opened or index == configured_source:
            cameras.append({"type": "camera", "value": index, "label": f"Camara {index}"})

    if not cameras:
        cameras.append({"type": "camera", "value": configured_source, "label": f"Camara {configured_source}"})

    return cameras


@app.route("/")
def index():
    return render_template(
        "index.html",
        detector_error=detector_error,
        reader_error=reader_error,
        video_aspect_ratio=Config.VIDEO_ASPECT_RATIO,
        video_max_width=Config.VIDEO_MAX_WIDTH,
    )


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
