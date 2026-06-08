import base64
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from flask import Flask, Response, redirect, render_template, request, url_for

from blueprints.api_bp import api_bp
from blueprints.video_bp import video_bp
from config import Config
from src.char_recognizer import CharRecognizer
from src.detector import PlateDetector
from src.incident_service import IncidentService
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


def image_data_url(image, ext=".png", quality=82):
    if image is None or getattr(image, "size", 0) == 0:
        return ""

    params = []
    mime = "image/png"
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        mime = "image/jpeg"

    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_shape_label(image):
    if image is None or getattr(image, "size", 0) == 0:
        return "--"
    h, w = image.shape[:2]
    return f"{w} x {h} px"


def make_ocr_step(title, description, image, used=True):
    return {
        "title": title,
        "description": description,
        "image_url": image_data_url(image),
        "shape": image_shape_label(image),
        "used": used,
    }


def load_debug_stage(debug_dir, stage):
    image_path = debug_dir / f"{stage}.png"
    if not image_path.exists():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def make_character_strip(char_images):
    if not char_images:
        return None

    cells = []
    for image in char_images:
        if image is None or getattr(image, "size", 0) == 0:
            continue
        if image.dtype in {np.float32, np.float64}:
            normalized = image
            if float(np.max(normalized)) <= 1.0:
                normalized = normalized * 255.0
            gray = np.clip(normalized, 0, 255).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        cells.append(cv2.resize(gray, (72, 72), interpolation=cv2.INTER_NEAREST))

    if not cells:
        return None

    gap = 10
    label_h = 22
    width = len(cells) * 72 + (len(cells) + 1) * gap
    height = 72 + label_h + gap * 2
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for index, cell in enumerate(cells):
        x = gap + index * (72 + gap)
        y = gap
        bgr = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        canvas[y:y + 72, x:x + 72] = bgr
        cv2.rectangle(canvas, (x, y), (x + 72, y + 72), (210, 220, 235), 1)
        cv2.putText(
            canvas,
            f"C{index + 1}",
            (x + 22, y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 23, 42),
            1,
            cv2.LINE_AA,
        )

    return canvas


OCR_STAGE_INFO = {
    "0_perspective_correction": (
        "Correccion de perspectiva",
        "Intenta enderezar la placa antes de segmentar caracteres.",
        True,
    ),
    "1_original": (
        "Recorte normalizado",
        "Placa redimensionada al ancho canonico usado por segmentacion.",
        True,
    ),
    "2_grayscale": (
        "Escala de grises",
        "Convierte el recorte a un solo canal para preparar binarizacion.",
        True,
    ),
    "3_clahe": (
        "Contraste local",
        "Mejora contraste por zonas para placas con iluminacion irregular.",
        False,
    ),
    "4_denoised": (
        "Suavizado",
        "Reduce ruido manteniendo bordes antes de separar caracteres.",
        True,
    ),
    "5_binary_selected": (
        "Binarizacion",
        "Mascara generada con el metodo configurado para buscar regiones de caracteres.",
        True,
    ),
    "7b_band_isolated": (
        "Banda de caracteres",
        "Conserva la franja principal de letras y numeros.",
        True,
    ),
    "7c_cleaned": (
        "Limpieza de ruido",
        "Mascara final antes de buscar candidatos por componentes.",
        True,
    ),
    "8_cc_bboxes": (
        "Candidatos por componentes",
        "Cajas iniciales encontradas por componentes conectados.",
        True,
    ),
    "10_final_bboxes": (
        "Bounding boxes de segmentacion",
        "Cajas validadas sobre la imagen normalizada del segmentador.",
        True,
    ),
    "11_chars_strip": (
        "Caracteres recortados",
        "Cada recorte individual que se envia a la CNN en orden izquierda a derecha.",
        True,
    ),
}


def detection_confidence_from_characters(characters):
    if not characters:
        return None
    confidences = [
        float(char.get("confidence", 0.0))
        for char in characters
        if char.get("confidence") is not None
    ]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def load_static_image(relative_path):
    if not relative_path:
        return None
    path = Path(relative_path)
    if not path.is_absolute():
        path = Config.STATIC_DIR / path
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def build_ocr_debug(crop):
    steps = []
    debug = None

    if crop is None or getattr(crop, "size", 0) == 0:
        return steps, debug

    steps.append(make_ocr_step(
        "Recorte recibido",
        "Recorte de placa usado para regenerar el diagnostico OCR.",
        crop,
        True,
    ))

    if plate_reader is None or not hasattr(plate_reader, "read_debug"):
        return steps, debug

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        debug_dir = Path(temp_dir) / "segmentacion"
        debug = plate_reader.read_debug(crop, debug_dir=debug_dir)

        for stage, (title, description, used) in OCR_STAGE_INFO.items():
            if stage == "11_chars_strip":
                image = make_character_strip(debug.get("char_images", []) if debug else [])
                if image is None:
                    image = load_debug_stage(debug_dir, stage)
            else:
                image = load_debug_stage(debug_dir, stage)
            if image is None:
                continue
            if stage == "5_binary_selected":
                method = (debug or {}).get("binarization_method", "otsu")
                method_label = "adaptativa" if method == "adaptive" else "Otsu"
                title = f"Binarizacion {method_label}"
            steps.append(make_ocr_step(title, description, image, used))

        bbox_image = debug.get("bbox_image") if debug else None
        if bbox_image is not None and getattr(bbox_image, "size", 0) > 0:
            steps.append(make_ocr_step(
                "Clasificacion de caracteres",
                "Cajas finales con letra o numero reconocido y confianza individual.",
                bbox_image,
                True,
            ))

    return steps, debug


def render_ocr_detail(crop, detection, active_page, back_url):
    steps, debug = build_ocr_debug(crop)
    debug_text = debug.get("text") if debug else ""
    debug_conf = debug.get("confidence") if debug else None

    display_detection = dict(detection or {})
    if debug is not None:
        display_detection["plate_text"] = debug_text
        display_detection["plate_text_confidence"] = debug_conf
        display_detection["characters"] = debug.get("characters") or []

    return render_template(
        "ocr_detalle.html",
        active_page=active_page,
        detection=display_detection,
        steps=steps,
        reader_error=reader_error,
        ocr_variants="CNN .pth",
        character_image_size=28,
        back_url=back_url,
    )


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
    crop = None
    if crop_bytes:
        crop = cv2.imdecode(np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    return render_ocr_detail(
        crop=crop,
        detection=detection,
        active_page="foto_radar",
        back_url=url_for("index"),
    )


@app.route("/incidencias/<incident_id>/ocr_detalle")
def incidencia_ocr_detalle(incident_id):
    incident = incident_service.get_incident(incident_id)
    if incident is None:
        return render_template("incidencia_no_encontrada.html", active_page="incidencias"), 404

    crop = load_static_image(incident.get("crop_path"))
    detection = {
        "plate_text": incident.get("plate_text"),
        "plate_text_confidence": detection_confidence_from_characters(
            incident.get("characters") or []
        ),
        "characters": incident.get("characters") or [],
    }
    return render_ocr_detail(
        crop=crop,
        detection=detection,
        active_page="incidencias",
        back_url=url_for("incidencia_detalle", incident_id=incident_id),
    )


@app.route("/actividad/<activity_id>/ocr_detalle")
def actividad_ocr_detalle(activity_id):
    item = incident_service.get_activity(activity_id)
    if item is None:
        return render_template("actividad_no_encontrada.html", active_page="actividad"), 404

    crop = load_static_image(item.get("crop_path"))
    detection = {
        "plate_text": item.get("plate_text"),
        "plate_text_confidence": item.get("plate_confidence"),
        "characters": item.get("characters") or [],
    }
    return render_ocr_detail(
        crop=crop,
        detection=detection,
        active_page="actividad",
        back_url=url_for("actividad_detalle", activity_id=activity_id),
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
