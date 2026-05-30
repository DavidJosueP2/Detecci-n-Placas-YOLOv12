import cv2
import numpy as np


BOX_COLOR = (255, 92, 0)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (255, 92, 0)
MUTED_COLOR = (148, 163, 184)
DISPLAY_LABEL = "License Plate"


def clamp(value, low, high):
    return max(low, min(value, high))


def process_frame(frame, detections):
    output = frame.copy()
    height, width = output.shape[:2]
    best_detection = choose_best_detection(detections)
    crops = []

    for detection in detections:
        x1, y1, x2, y2 = clip_box(detection, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOR, 2, cv2.LINE_AA)
        draw_label(output, detection, x1, y1, width)

        crop = frame[y1:y2, x1:x2].copy()
        crops.append({"detection": detection, "crop": crop})

    if not detections:
        draw_corner_status(output, "Sin placa detectada")

    crops.sort(key=lambda item: item["detection"]["confidence"], reverse=True)
    return output, crops, best_detection


def choose_best_detection(detections):
    if not detections:
        return None

    tracked = [item for item in detections if item.get("track_id") is not None]
    if tracked:
        return max(tracked, key=lambda item: item["confidence"])

    return max(detections, key=lambda item: item["confidence"])


def clip_box(detection, width, height):
    x1 = clamp(int(detection["x1"]), 0, width - 1)
    y1 = clamp(int(detection["y1"]), 0, height - 1)
    x2 = clamp(int(detection["x2"]), 0, width - 1)
    y2 = clamp(int(detection["y2"]), 0, height - 1)
    return x1, y1, x2, y2


def draw_label(image, detection, x1, y1, width):
    label = f'{DISPLAY_LABEL} {detection["confidence"]:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    thickness = 1
    padding_x = 6
    padding_y = 4

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_w = text_w + padding_x * 2
    label_h = text_h + baseline + padding_y * 2
    lx1 = clamp(x1, 0, max(0, width - label_w - 1))
    ly1 = y1 - label_h if y1 - label_h >= 0 else y1
    lx2 = lx1 + label_w
    ly2 = ly1 + label_h

    cv2.rectangle(image, (lx1, ly1), (lx2, ly2), PANEL_COLOR, -1)
    cv2.putText(
        image,
        label,
        (lx1 + padding_x, ly2 - baseline - padding_y),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_corner_status(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (18, 30), font, 0.55, MUTED_COLOR, 1, cv2.LINE_AA)


def encode_jpeg(image):
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not success:
        return None
    return buffer.tobytes()


def placeholder_frame(message, width=960, height=540):
    frame = np.full((height, width, 3), (246, 248, 251), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (191, 200, 214), 1)
    cv2.putText(
        frame,
        message,
        (36, height // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (93, 104, 120),
        1,
        cv2.LINE_AA,
    )
    return frame


def placeholder_crop():
    crop = np.full((160, 360, 3), (246, 248, 251), dtype=np.uint8)
    cv2.rectangle(crop, (0, 0), (359, 159), (191, 200, 214), 1)
    cv2.putText(
        crop,
        "Sin recorte",
        (92, 89),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (93, 104, 120),
        1,
        cv2.LINE_AA,
    )
    return crop
