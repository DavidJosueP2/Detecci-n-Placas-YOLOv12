from pathlib import Path
import argparse
import os
import sys

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "Best_epoch_2_90mil.pt"
FALLBACK_MODEL_PATH = ROOT / "Best_epoch_1_90mil.pt"
OUTPUT_DIR = ROOT / "resultados_individuales"
BOX_COLOR = (255, 70, 0)
TEXT_COLOR = (255, 255, 255)


def clamp(value, low, high):
    return max(low, min(value, high))


def choose_image():
    try:
        from tkinter import Tk, filedialog
    except Exception:
        return None

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Elige una imagen de placa",
        filetypes=[
            ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Todos", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def open_image(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


def find_plate_mask(image, x1, y1, x2, y2):
    height, width = image.shape[:2]
    x1 = clamp(x1, 0, width - 1)
    x2 = clamp(x2, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    y2 = clamp(y2, 0, height - 1)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    white_plate = ((v > 125) & (s < 95)).astype("uint8") * 255
    yellow_plate = ((h >= 10) & (h <= 45) & (s > 20) & (v > 85)).astype("uint8") * 255
    color_mask = cv2.bitwise_or(white_plate, yellow_plate)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, max(80, int(gray.mean() + gray.std() * 0.25)), 255, cv2.THRESH_BINARY)
    search_mask = cv2.bitwise_or(color_mask, bright_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, crop.shape[1] // 22), 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = crop.shape[0] * crop.shape[1]
    best_component_mask = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < crop_area * 0.06:
            continue

        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / short_side
        if not 1.25 <= aspect <= 5.8:
            continue

        rect_area = rw * rh
        if rect_area > crop_area * 0.72:
            continue

        component_mask = color_mask.copy() * 0
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        color_inside = cv2.mean(color_mask, mask=component_mask)[0] / 255
        if color_inside < 0.18:
            continue

        rectangularity = area / rect_area
        aspect_bonus = 1 / (1 + abs(aspect - 2.4))
        score = area * rectangularity * (0.6 + color_inside) * (0.7 + aspect_bonus)
        if score > best_score:
            best_score = score
            best_component_mask = component_mask

    full_mask = None
    if best_component_mask is not None:
        precise_mask = cv2.bitwise_and(search_mask, best_component_mask)
        precise_mask = cv2.morphologyEx(precise_mask, cv2.MORPH_CLOSE, open_kernel, iterations=1)
        full_mask = image[:, :, 0] * 0
        full_mask[y1:y2, x1:x2] = precise_mask

    return full_mask


def draw_detection_style(image_path, result, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    image_height, image_width = image.shape[:2]
    base_size = min(image_width, image_height)
    thickness = max(2, min(4, round(base_size / 420)))
    font_scale = max(0.45, min(0.75, base_size / 1050))
    font_thickness = max(1, thickness - 1)
    padding = max(3, thickness + 2)

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names.get(cls_id, str(cls_id))
        label = f"{str(name).replace('_', ' ')} {conf:.2f}"

        x1 = clamp(x1, 0, image_width - 1)
        x2 = clamp(x2, 0, image_width - 1)
        y1 = clamp(y1, 0, image_height - 1)
        y2 = clamp(y2, 0, image_height - 1)

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        label_x1 = x1
        label_y1 = y1 - text_h - baseline - padding * 2
        if label_y1 < 0:
            label_y1 = y1

        label_x2 = min(label_x1 + text_w + padding * 2, image_width - 1)
        label_y2 = min(label_y1 + text_h + baseline + padding * 2, image_height - 1)
        cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), BOX_COLOR, -1)
        cv2.putText(
            image,
            label,
            (label_x1 + padding, label_y2 - baseline - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            font_thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Prueba un modelo YOLO para detectar placas.")
    parser.add_argument("image", nargs="?", help="Ruta de la imagen a probar.")
    parser.add_argument(
        "--model",
        default=None,
        help="Ruta del modelo .pt. Por defecto usa best_placas_ecuador.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image) if args.image else choose_image()
    if not image_path:
        print("No se eligio ninguna imagen.")
        return 1

    image_path = image_path.resolve()
    if not image_path.exists():
        print(f"No existe la imagen: {image_path}")
        return 1

    model_path = Path(args.model).resolve() if args.model else DEFAULT_MODEL_PATH
    if not model_path.exists() and not args.model:
        model_path = FALLBACK_MODEL_PATH

    if not model_path.exists():
        print(f"No existe el modelo: {model_path}")
        return 1

    print(f"Modelo usado: {model_path}")
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        save=False,
        verbose=False,
    )

    final_file = OUTPUT_DIR / f"{image_path.stem}_detectado{image_path.suffix}"

    OUTPUT_DIR.mkdir(exist_ok=True)
    if draw_detection_style(image_path, results[0], final_file):
        print(f"Resultado guardado en: {final_file}")
        open_image(str(final_file))
    else:
        plotted = results[0].plot()
        cv2.imwrite(str(final_file), plotted)
        print(f"Resultado guardado en: {final_file}")
        open_image(str(final_file))

    boxes = results[0].boxes
    print(f"Detecciones: {len(boxes)}")
    for i, box in enumerate(boxes, start=1):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names.get(cls_id, str(cls_id))
        print(f"{i}. {name} - confianza {conf:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
