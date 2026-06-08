#!/usr/bin/env python3
"""
Extrae el recorte de la placa de una imagen y lo guarda.
Uso: python extract_plate_crop.py --image ruta/a/coche.jpg
"""

import sys
from pathlib import Path
import argparse
import cv2

# Añadir ruta del proyecto para importar PlateDetector
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Ajusta según tu estructura
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from plate_detection_web.src.detector import PlateDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a la imagen del vehículo")
    parser.add_argument("--model", default=None, help="Ruta al modelo .pt (opcional)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confianza mínima")
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta para guardar el recorte (por defecto: placa_recortada.jpg)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return 1

    # Cargar modelo
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = PROJECT_ROOT / "models" / "plate_detection" / "90mil_50 epocas.pt"
        if not model_path.exists():
            candidates = list(
                (PROJECT_ROOT / "models" / "plate_detection").glob("*.pt")
            )
            if candidates:
                model_path = candidates[0]
                print(f"⚠️ Usando modelo alternativo: {model_path.name}")
            else:
                print("❌ No se encontró modelo en models/plate_detection/")
                return 1

    detector = PlateDetector(
        model_path=str(model_path), confidence_threshold=args.conf, device="cpu"
    )

    img = cv2.imread(str(image_path))
    if img is None:
        print("❌ No se pudo leer la imagen")
        return 1

    detections = detector.detect(img)
    if not detections:
        print("❌ No se detectó ninguna placa")
        return 1

    # Tomar la detección de mayor confianza
    best = max(detections, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]

    # Asegurar límites
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    plate_crop = img[y1:y2, x1:x2]
    if plate_crop.size == 0:
        print("❌ Recorte vacío")
        return 1

    # Guardar recorte
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = image_path.parent / f"{image_path.stem}_plate_crop.jpg"

    cv2.imwrite(str(out_path), plate_crop)
    print(f"✅ Recorte de placa guardado en: {out_path}")
    print(f"   Dimensiones: {plate_crop.shape[1]}x{plate_crop.shape[0]} píxeles")

    return 0


if __name__ == "__main__":
    sys.exit(main())
