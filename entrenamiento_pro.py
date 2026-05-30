from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos"

# Cambia este valor si el dataset grande queda en otro YAML.
DEFAULT_DATA_YAML = ROOT / "dataset_grande.yaml"


AUGMENTACION_PLACAS_PRO = {
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.30,
    "degrees": 7.0,
    "translate": 0.10,
    "scale": 0.45,
    "shear": 2.5,
    "perspective": 0.0007,
    "fliplr": 0.10,
    "flipud": 0.0,
    "mosaic": 0.65,
    "mixup": 0.03,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento pro para deteccion de placas con YOLO12 preentrenado."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Ruta al data.yaml del dataset.")
    parser.add_argument("--model", default="yolo12s.pt", help="Modelo base preentrenado: yolo12n.pt/s.pt/m.pt.")
    parser.add_argument("--epochs", type=int, default=50, help="Epocas recomendadas para dataset grande.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping si validacion no mejora.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen. 640 es seguro para GPU de 4 GB.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size. Para 4 GB empieza con 2 o 4.")
    parser.add_argument("--workers", type=int, default=0, help="0 evita errores de multiprocessing en Windows.")
    parser.add_argument("--device", default="auto", help="auto, cpu o 0 para GPU CUDA.")
    parser.add_argument("--name", default="placas_yolo12_pro", help="Nombre del experimento.")
    parser.add_argument("--cache", default=False, help="False, ram o disk. Para 90k suele convenir disk.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraccion del train a usar. 0.01 sirve para smoke test.")
    parser.add_argument("--multi-scale", type=float, default=0.0, help="Rango multi-scale. 0 es mas estable en GPU pequena.")
    parser.add_argument("--resume", action="store_true", help="Continuar desde modelos/<name>/weights/last.pt.")
    parser.add_argument("--sin-augment", action="store_true", help="Desactiva aumentacion para comparar.")
    parser.add_argument("--quick", action="store_true", help="Prueba rapida de 1 epoca.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"No existe el YAML del dataset: {data_yaml}")
        print("Cuando tengas el dataset grande, crea su data.yaml y pasalo con --data.")
        return 1

    device = resolve_device(args.device)
    epochs = 1 if args.quick else args.epochs
    augmentacion = {} if args.sin_augment else AUGMENTACION_PLACAS_PRO

    print(f"Dataset: {data_yaml}")
    resume_path = MODELOS_DIR / args.name / "weights" / "last.pt"
    model_source = resume_path if args.resume else args.model

    if args.resume and not resume_path.exists():
        print(f"No existe checkpoint para reanudar: {resume_path}")
        return 1

    print(f"Modelo: {model_source}")
    print(f"Dispositivo: {device}")
    print(f"Epocas: {epochs}")
    print(f"Batch: {args.batch}")
    print(f"Fraccion train: {args.fraction}")
    print("Aumentacion:", "desactivada" if args.sin_augment else augmentacion)

    modelo = YOLO(str(model_source))
    modelo.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        pretrained=True,
        project=str(MODELOS_DIR),
        name=args.name,
        exist_ok=True,
        resume=True if args.resume else False,
        seed=42,
        deterministic=False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        close_mosaic=10,
        multi_scale=args.multi_scale,
        cache=args.cache,
        fraction=args.fraction,
        plots=True,
        val=True,
        **augmentacion,
    )

    metricas = modelo.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(MODELOS_DIR),
        name=f"{args.name}_val",
        exist_ok=True,
    )
    print(metricas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
