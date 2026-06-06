import base64

import cv2
import numpy as np
from flask import Flask, Response, redirect, render_template, request, url_for

from blueprints.api_bp import api_bp
from blueprints.video_bp import video_bp
from config import Config
from src.detector import PlateDetector
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

if Config.USE_CNN_RECOGNIZER:
    from src.char_recognizer import CharRecognizer
    try:
        plate_reader = CharRecognizer(
            model_path=str(Config.CNN_MODEL_PATH),
            device=Config.DEVICE,
            crop_source=Config.CNN_CROP_SOURCE,
        )
        print(f"[OCR] CharRecognizer cargado: {Config.CNN_MODEL_PATH}")
    except Exception as exc:
        plate_reader = None
        reader_error = str(exc)
        print(f"[OCR] CharRecognizer falló, sin OCR: {exc}")
else:
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
    stream_jpeg_quality=Config.STREAM_JPEG_QUALITY,
    detection_every_n_frames=Config.DETECTION_EVERY_N_FRAMES,
    live_detection_interval_seconds=Config.LIVE_DETECTION_INTERVAL_SECONDS,
    live_detection_mode=Config.LIVE_DETECTION_MODE,
    ocr_interval_seconds=Config.OCR_INTERVAL_SECONDS,
    ocr_retry_interval_seconds=Config.OCR_RETRY_INTERVAL_SECONDS,
    ocr_max_plates_per_frame=Config.OCR_MAX_PLATES_PER_FRAME,
    ocr_min_detection_confidence=Config.OCR_MIN_DETECTION_CONFIDENCE,
    ocr_zone_x1=Config.OCR_ZONE_X1,
    ocr_zone_y1=Config.OCR_ZONE_Y1,
    ocr_zone_x2=Config.OCR_ZONE_X2,
    ocr_zone_y2=Config.OCR_ZONE_Y2,
    ocr_zone_min_overlap=Config.OCR_ZONE_MIN_OVERLAP,
    camera_backend=Config.CAMERA_BACKEND,
    camera_width=Config.CAMERA_WIDTH,
    camera_height=Config.CAMERA_HEIGHT,
    camera_fps=Config.CAMERA_FPS,
    camera_fourcc=Config.CAMERA_FOURCC,
    incident_service=incident_service,
    detector_error=detector_error or reader_error,
)

app.stream = stream
app.incident_service = incident_service

app.register_blueprint(video_bp)
app.register_blueprint(api_bp)


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


@app.route("/actividad")
def actividad():
    return render_template(
        "actividad.html",
        activity=incident_service.list_activity(),
        active_page="actividad",
    )


@app.route("/actividad/<activity_id>")
def actividad_detalle(activity_id):
    item = incident_service.get_activity(activity_id)
    if item is None:
        return render_template("actividad_no_encontrada.html", active_page="actividad"), 404
    return render_template(
        "actividad_detalle.html",
        item=item,
        active_page="actividad",
        email_status=request.args.get("email_status", ""),
    )


@app.route("/actividad/<activity_id>/enviar_correo", methods=["POST"])
def enviar_correo_actividad(activity_id):
    result = incident_service.send_activity_email(activity_id)
    if result.get("already_sent"):
        status = "already"
    elif result.get("ok"):
        status = "sent"
    else:
        status = "error"

    return redirect(url_for("actividad_detalle", activity_id=activity_id, email_status=status))


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


@app.route("/plate_crop")
def plate_crop():
    return Response(stream.current_crop(), mimetype="image/jpeg")


@app.route("/current_frame")
def current_frame():
    return Response(stream.current_frame(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False,
    )
