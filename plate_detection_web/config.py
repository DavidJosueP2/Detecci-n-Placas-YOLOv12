import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def load_env_file(path):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = clean_env_value(value)
        if key and key not in os.environ:
            os.environ[key] = value


def clean_env_value(value):
    value = value.strip()
    quote = None

    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue

        if char == "#" and quote is None:
            value = value[:index].strip()
            break

    return value.strip().strip('"').strip("'")


load_env_file(PROJECT_ROOT / ".env")
load_env_file(BASE_DIR / ".env")


def parse_video_source(value):
    if isinstance(value, int):
        return value

    value = str(value).strip()
    if value.isdigit():
        return int(value)

    return value


def resolve_model_path():
    configured = os.getenv("PLATE_MODEL_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path

    candidates = [
        BASE_DIR / "models" / "90mil_50 epocas.pt",
        
        #BASE_DIR / "models" / "90mil_10 epocas.pt",
        #BASE_DIR / "models" / "lp_yolo11x_morsetech.pt",
        #PROJECT_ROOT / "models" / "Best_epoch_2_90mil.pt",
        #PROJECT_ROOT / "models" / "best_placas_ecuador.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def resolve_character_model_path():
    configured = os.getenv("CHARACTER_MODEL_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path

    candidates = [
        BASE_DIR / "models" / "Character-LP.pt",
        BASE_DIR / "models" / "Charcter-LP.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted((BASE_DIR / "models").glob("*Char*LP*.pt"))
    if matches:
        return matches[0]

    return candidates[0]


class Config:
    MODEL_PATH = resolve_model_path()
    CHARACTER_MODEL_PATH = resolve_character_model_path()
    VIDEO_SOURCE = parse_video_source(os.getenv("VIDEO_SOURCE", "0"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.12"))
    CHARACTER_CONFIDENCE_THRESHOLD = float(os.getenv("CHARACTER_CONFIDENCE_THRESHOLD", "0.25"))
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
    CHARACTER_IMAGE_SIZE = int(os.getenv("CHARACTER_IMAGE_SIZE", "320"))
    OCR_VARIANTS = int(os.getenv("OCR_VARIANTS", "4"))
    DEVICE = os.getenv("YOLO_DEVICE", "auto")
    TRACKER = os.getenv("YOLO_TRACKER", "botsort.yaml")
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    SPEED_LINE_A_Y = float(
        os.getenv("SPEED_LINE_A_LEFT_Y", os.getenv("SPEED_LINE_A_Y", "0.45"))
    )
    SPEED_LINE_A_RIGHT_Y = float(
        os.getenv("SPEED_LINE_A_RIGHT_Y", os.getenv("SPEED_LINE_A_Y", "0.45"))
    )
    SPEED_LINE_B_Y = float(
        os.getenv("SPEED_LINE_B_LEFT_Y", os.getenv("SPEED_LINE_B_Y", "0.70"))
    )
    SPEED_LINE_B_RIGHT_Y = float(
        os.getenv("SPEED_LINE_B_RIGHT_Y", os.getenv("SPEED_LINE_B_Y", "0.70"))
    )
    SPEED_ROI_X1 = float(os.getenv("SPEED_ROI_X1", "0.0"))
    SPEED_ROI_X2 = float(os.getenv("SPEED_ROI_X2", "1.0"))
    SPEED_DISTANCE_METERS = float(os.getenv("SPEED_DISTANCE_METERS", "5.0"))
    SPEED_DIRECTION = os.getenv("SPEED_DIRECTION", "both")
    SPEED_LINE_HYSTERESIS_PX = float(os.getenv("SPEED_LINE_HYSTERESIS_PX", "8"))
    SPEED_MIN_TRAVEL_TIME = float(os.getenv("SPEED_MIN_TRAVEL_TIME", "0.15"))
    SPEED_MAX_TRAVEL_TIME = float(os.getenv("SPEED_MAX_TRAVEL_TIME", "8.0"))
    SPEED_MIN_MOVEMENT_PX = float(os.getenv("SPEED_MIN_MOVEMENT_PX", "35"))
    SPEED_MIN_PARTIAL_PROGRESS = float(os.getenv("SPEED_MIN_PARTIAL_PROGRESS", "0.35"))
    STREAM_TARGET_FPS = float(os.getenv("STREAM_TARGET_FPS", "12"))
    STREAM_MAX_WIDTH = int(os.getenv("STREAM_MAX_WIDTH", "960"))
    DETECTION_EVERY_N_FRAMES = int(os.getenv("DETECTION_EVERY_N_FRAMES", "6"))
    LIVE_DETECTION_INTERVAL_SECONDS = float(os.getenv("LIVE_DETECTION_INTERVAL_SECONDS", "0.0"))
    LIVE_DETECTION_MODE = os.getenv("LIVE_DETECTION_MODE", "detect")
    OCR_INTERVAL_SECONDS = float(os.getenv("OCR_INTERVAL_SECONDS", "1.25"))
    OCR_RETRY_INTERVAL_SECONDS = float(os.getenv("OCR_RETRY_INTERVAL_SECONDS", "0.30"))
    OCR_MAX_PLATES_PER_FRAME = int(os.getenv("OCR_MAX_PLATES_PER_FRAME", "1"))
    OCR_MIN_DETECTION_CONFIDENCE = float(os.getenv("OCR_MIN_DETECTION_CONFIDENCE", "0.20"))
    # Solo controla como se dibuja el visor en la web; no cambia el video real.
    VIDEO_ASPECT_RATIO = os.getenv("VIDEO_ASPECT_RATIO", "16 / 9")
    VIDEO_MAX_WIDTH = os.getenv("VIDEO_MAX_WIDTH", "860px")
    CAMERA_SCAN_LIMIT = int(os.getenv("CAMERA_SCAN_LIMIT", "10"))
    CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "msmf")
    CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
    CAMERA_FPS = float(os.getenv("CAMERA_FPS", "30"))
    CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG")
    UPLOAD_DIR = BASE_DIR / "static" / "captures"
    STATIC_DIR = BASE_DIR / "static"
    DATA_DIR = BASE_DIR / "data"
    INCIDENT_DB_PATH = DATA_DIR / os.getenv("INCIDENT_DB_NAME", "incidencias.sqlite3")
    INCIDENT_COOLDOWN_SECONDS = float(os.getenv("INCIDENT_COOLDOWN_SECONDS", "45"))
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
