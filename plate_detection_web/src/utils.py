from datetime import datetime


def now_label():
    return datetime.now().strftime("%H:%M:%S")


def detection_status(detection):
    if not detection:
        return {
            "detected": False,
            "confidence": 0.0,
            "class_id": None,
            "class_name": None,
            "track_id": None,
            "detections": [],
            "plate_text": "",
            "plate_text_confidence": 0.0,
            "characters": [],
            "speed_kmh": None,
            "speed_status": "esperando cruce",
            "message": "Sin deteccion",
            "timestamp": now_label(),
        }

    return {
        "detected": True,
        "confidence": detection["confidence"],
        "class_id": detection["class_id"],
        "class_name": "License Plate",
        "track_id": detection.get("track_id"),
        "detections": [],
        "plate_text": detection.get("plate_text") or "",
        "plate_text_confidence": detection.get("plate_text_confidence", 0.0),
        "characters": detection.get("characters", []),
        "speed_kmh": detection.get("speed_kmh"),
        "speed_status": detection.get("speed_status", "esperando cruce"),
        "message": "Placa detectada",
        "timestamp": now_label(),
    }
