from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos"

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


AUGMENTACION_PLACAS_SUAVE = {
    "hsv_h": 0.005,
    "hsv_s": 0.10,
    "hsv_v": 0.10,
    "degrees": 2.0,
    "translate": 0.03,
    "scale": 0.15,
    "shear": 0.5,
    "perspective": 0.0001,
    "fliplr": 0.05,
    "flipud": 0.0,
    "mosaic": 0.10,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento/fine-tuning para deteccion de placas con YOLO."
    )

    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Ruta al data.yaml del dataset.")
    parser.add_argument("--model", default="yolo12s.pt", help="Modelo base o checkpoint .pt.")
    parser.add_argument("--epochs", type=int, default=30, help="Epocas de entrenamiento.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping si validacion no mejora.")
    parser.add_argument("--imgsz", type=int, default=768, help="Tamano de imagen.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Workers para carga de datos.")
    parser.add_argument("--device", default="auto", help="auto, cpu o 0 para GPU CUDA.")
    parser.add_argument("--name", default="placas_finetune", help="Nombre del experimento.")
    parser.add_argument("--cache", default=False, help="False, ram o disk.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraccion del train a usar.")
    parser.add_argument("--multi-scale", type=float, default=0.0, help="Rango multi-scale.")
    parser.add_argument("--resume", action="store_true", help="Continuar desde modelos/<name>/weights/last.pt.")
    parser.add_argument("--quick", action="store_true", help="Prueba rapida de 1 epoca.")

    parser.add_argument("--lr0", type=float, default=0.0001, help="Learning rate inicial.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Learning rate final relativo.")
    parser.add_argument(
        "--aug",
        default="suave",
        choices=["suave", "pro", "none"],
        help="Tipo de data augmentation: suave, pro o none."
    )

    parser.add_argument("--sin-augment", action="store_true", help="Desactiva aumentacion.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_augmentacion(args):
    if args.sin_augment or args.aug == "none":
        return {}

    if args.aug == "pro":
        return AUGMENTACION_PLACAS_PRO

    return AUGMENTACION_PLACAS_SUAVE


def main():
    args = parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"No existe el YAML del dataset: {data_yaml}")
        return 1

    device = resolve_device(args.device)
    epochs = 1 if args.quick else args.epochs
    augmentacion = resolve_augmentacion(args)

    resume_path = MODELOS_DIR / args.name / "weights" / "last.pt"
    model_source = resume_path if args.resume else Path(args.model)

    if args.resume and not resume_path.exists():
        print(f"No existe checkpoint para reanudar: {resume_path}")
        return 1

    if not args.resume and not Path(model_source).exists() and str(model_source).endswith((".pt", ".pk")):
        print(f"No existe el modelo indicado: {model_source}")
        return 1

    print("============================================================")
    print("CONFIGURACION DE ENTRENAMIENTO")
    print("============================================================")
    print(f"Dataset: {data_yaml}")
    print(f"Modelo: {model_source}")
    print(f"Dispositivo: {device}")
    print(f"Epocas: {epochs}")
    print(f"Patience: {args.patience}")
    print(f"Batch: {args.batch}")
    print(f"Workers: {args.workers}")
    print(f"Imagen: {args.imgsz}")
    print(f"LR0: {args.lr0}")
    print(f"LRF: {args.lrf}")
    print(f"Cache: {args.cache}")
    print(f"Fraccion train: {args.fraction}")
    print(f"Augmentacion seleccionada: {'desactivada' if not augmentacion else args.aug}")
    print(f"Aumentacion: {augmentacion if augmentacion else 'desactivada'}")
    print("============================================================")

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
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=2,
        weight_decay=0.0005,
        close_mosaic=5,
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