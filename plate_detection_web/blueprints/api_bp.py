from flask import Blueprint, current_app, jsonify, request

from config import Config
from src.char_recognizer import current_binarization_config, update_binarization_config
from src.fuzzy_mamdani import get_mamdani_config, set_mamdani_config

api_bp = Blueprint("api", __name__, url_prefix="/api")

SPEED_CONFIG_ENV_KEYS = {
    "line_a_left_y": "SPEED_LINE_A_LEFT_Y",
    "line_a_right_y": "SPEED_LINE_A_RIGHT_Y",
    "line_b_left_y": "SPEED_LINE_B_LEFT_Y",
    "line_b_right_y": "SPEED_LINE_B_RIGHT_Y",
    "roi_x1": "SPEED_ROI_X1",
    "roi_x2": "SPEED_ROI_X2",
    "distance_meters": "SPEED_DISTANCE_METERS",
}


def update_env_values(values):
    env_path = Config.BASE_DIR / ".env"
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


@api_bp.route("/speed_config", methods=["GET", "POST"])
def speed_config():
    if request.method == "GET":
        return jsonify({"ok": True, "config": current_app.stream.current_speed_config()})

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

    updated = current_app.stream.update_speed_config(**accepted)
    env_values = {
        env_key: f"{updated[field]:.4f}".rstrip("0").rstrip(".")
        for field, env_key in SPEED_CONFIG_ENV_KEYS.items()
        if field in accepted
    }
    if env_values:
        update_env_values(env_values)

    return jsonify({"ok": True, "config": updated})


@api_bp.route("/ocr_zone_config", methods=["GET", "POST"])
def ocr_zone_config():
    if request.method == "GET":
        return jsonify({"ok": True, "config": current_app.stream.current_ocr_zone_config()})

    payload = request.get_json(silent=True) or {}
    accepted = {}
    for field in {"x1", "y1", "x2", "y2"}:
        if field not in payload:
            continue
        try:
            accepted[field] = float(payload[field])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": f"Valor invalido: {field}"}), 400

    updated = current_app.stream.update_ocr_zone_config(**accepted)
    return jsonify({"ok": True, "config": updated})


@api_bp.route("/ocr_binarization_config", methods=["GET", "POST"])
def ocr_binarization_config():
    if request.method == "GET":
        return jsonify({"ok": True, "config": current_binarization_config()})

    payload = request.get_json(silent=True) or {}
    try:
        updated = update_binarization_config(payload.get("method", "otsu"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "config": updated})


@api_bp.route("/mamdani_config", methods=["GET", "POST"])
def mamdani_config():
    if request.method == "GET":
        return jsonify({"ok": True, "config": get_mamdani_config()})

    payload = request.get_json(silent=True) or {}
    try:
        updated = set_mamdani_config(payload)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "config": updated})


@api_bp.route("/email_config", methods=["GET", "POST"])
def email_config():
    if request.method == "GET":
        current = current_app.incident_service.current_email_config()
        return jsonify({"ok": True, "config": {"enabled": current["enabled"]}})

    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))
    updated = current_app.incident_service.update_email_config(enabled=enabled)
    update_env_values({"EMAIL_INCIDENTS_ENABLED": "1" if updated["enabled"] else "0"})
    return jsonify({"ok": True, "config": {"enabled": updated["enabled"]}})
