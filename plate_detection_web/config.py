import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


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
        return Path(configured)

    candidates = [
        BASE_DIR / "models" / "90mil_10 epocas.pt",
        BASE_DIR / "models" / "lp_yolo11x_morsetech.pt",
        PROJECT_ROOT / "models" / "Best_epoch_2_90mil.pt",
        PROJECT_ROOT / "models" / "best_placas_ecuador.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def resolve_character_model_path():
    configured = os.getenv("CHARACTER_MODEL_PATH")
    if configured:
        return Path(configured)

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
    DEVICE = os.getenv("YOLO_DEVICE", "auto")
    TRACKER = os.getenv("YOLO_TRACKER", "botsort.yaml")
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    SPEED_PIXELS_PER_METER = float(os.getenv("SPEED_PIXELS_PER_METER", "45"))
    SPEED_SMOOTHING = float(os.getenv("SPEED_SMOOTHING", "0.35"))
    # Solo controla como se dibuja el visor en la web; no cambia el video real.
    VIDEO_ASPECT_RATIO = os.getenv("VIDEO_ASPECT_RATIO", "16 / 9")
    VIDEO_MAX_WIDTH = os.getenv("VIDEO_MAX_WIDTH", "860px")
    CAMERA_SCAN_LIMIT = int(os.getenv("CAMERA_SCAN_LIMIT", "4"))
    UPLOAD_DIR = BASE_DIR / "static" / "captures"
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
