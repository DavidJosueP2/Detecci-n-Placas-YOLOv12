import cv2
import numpy as np


BOX_COLOR = (255, 92, 0)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (255, 92, 0)
MUTED_COLOR = (148, 163, 184)
LINE_A_COLOR = (0, 196, 255)
LINE_B_COLOR = (56, 217, 140)
DISPLAY_LABEL = "License Plate"


def clamp(value, low, high):
    return max(low, min(value, high))


def process_frame(frame, detections, speed_lines=None, stats=None):
    output = frame.copy()
    height, width = output.shape[:2]
    best_detection = choose_best_detection(detections)
    crops = []

    draw_speed_lines(output, speed_lines or [])

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

    draw_stats_overlay(output, stats or {})

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


def draw_speed_lines(image, speed_lines):
    if not speed_lines:
        return

    height, width = image.shape[:2]
    colors = {"A": LINE_A_COLOR, "B": LINE_B_COLOR}
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in speed_lines:
        x1 = clamp(int(line.get("x1", 0)), 0, width - 1)
        x2 = clamp(int(line.get("x2", width - 1)), 0, width - 1)
        y1 = clamp(int(line.get("y1", line.get("y", 0))), 0, height - 1)
        y2 = clamp(int(line.get("y2", line.get("y", 0))), 0, height - 1)
        name = line.get("name", "")
        color = colors.get(name, (255, 255, 255))
        label = line.get("label", f"Linea {name}")
        cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (min(width - 120, x1 + 16), max(22, y1 - 8)),
            font,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_stats_overlay(image, stats):
    if not stats:
        return

    rows = [
        f"FPS {stats.get('fps', 0.0):.1f}",
        f"YOLO {stats.get('detector_ms', 0.0):.0f} ms",
        f"Placas {stats.get('detections', 0)}",
    ]
    source_fps = stats.get("source_fps", 0.0)
    if source_fps:
        rows.append(f"Video {source_fps:.1f} fps")
    speed_status = stats.get("speed_status")
    if speed_status:
        rows.append(str(speed_status)[:24])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    line_height = 22
    padding = 10
    width = 178
    height = padding * 2 + line_height * len(rows)
    x2 = image.shape[1] - 12
    y1 = 12
    x1 = max(8, x2 - width)
    y2 = y1 + height

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (2, 6, 23), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 41, 59), 1, cv2.LINE_AA)

    for index, row in enumerate(rows):
        y = y1 + padding + 15 + index * line_height
        cv2.putText(image, row, (x1 + padding, y), font, font_scale, (226, 232, 240), 1, cv2.LINE_AA)


def encode_jpeg(image, quality=68):
    quality = int(clamp(quality, 35, 95))
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return buffer.tobytes()


def resize_to_max_width(image, max_width):
    if not max_width or max_width <= 0:
        return image

    height, width = image.shape[:2]
    if width <= max_width:
        return image

    scale = max_width / width
    size = (max_width, max(1, int(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


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
