from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "placas_ecuador_preparado.yaml"
MODELOS_DIR = ROOT / "modelos"

# Importante: .yaml = arquitectura, pesos aleatorios.
# No usar .pt aqui, porque .pt puede traer pesos preentrenados.
ARQUITECTURA = "yolo12.yaml"

# Aumentacion online: YOLO aplica estas transformaciones durante el entrenamiento.
# No creamos copias fisicas porque eso infla el dataset y puede meter duplicados.
# Para placas conviene ser moderado: rotar/perspectiva si, pero no deformar tanto que
# la placa deje de parecer una placa real ecuatoriana.
AUGMENTACION_PLACAS = {
    "hsv_h": 0.01,          # cambios leves de tono
    "hsv_s": 0.35,         # placas con diferente saturacion/camaras
    "hsv_v": 0.30,         # sol, sombra, noche parcial, exposicion
    "degrees": 8.0,        # inclinaciones reales por angulo de camara
    "translate": 0.10,     # placa no siempre centrada
    "scale": 0.45,         # placas cercanas y lejanas
    "shear": 3.0,          # ligera deformacion por perspectiva
    "perspective": 0.0008, # perspectiva suave; no exagerar en bbox normales
    "fliplr": 0.15,        # poco, porque texto espejado no es muy realista
    "flipud": 0.0,         # nunca placas de cabeza en el caso normal
    "mosaic": 0.75,        # ayuda con dataset pequeno y objetos en contexto
    "mixup": 0.0,          # evita placas fantasma/transparentes
    "cutmix": 0.0,         # evita recortes artificiales poco realistas
    "copy_paste": 0.0,     # solo tiene sentido fuerte en segmentacion
}


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar detector de placas con YOLO12 desde cero.")
    parser.add_argument("--epochs", type=int, default=80, help="Numero de epocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen para entrenamiento.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size. Usa -1 para auto-batch.")
    parser.add_argument("--workers", type=int, default=0, help="Workers del dataloader. 0 es mas estable en Windows.")
    parser.add_argument("--name", default="placas_yolo12_scratch", help="Nombre de carpeta del experimento.")
    parser.add_argument("--device", default="auto", help="auto, cpu o 0 si tienes GPU CUDA.")
    parser.add_argument("--quick", action="store_true", help="Prueba rapida de 1 epoca.")
    parser.add_argument("--sin-augment", action="store_true", help="Desactiva aumentacion para comparar.")
    return parser.parse_args()


def entrenar():
    args = parse_args()
    epochs = 1 if args.quick else args.epochs
    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    modelo = YOLO(ARQUITECTURA)
    augmentacion = {} if args.sin_augment else AUGMENTACION_PLACAS

    print("Entrenamiento desde cero: yolo12.yaml + pretrained=False")
    print(f"Dataset: {DATA_YAML}")
    print(f"Dispositivo: {device}")
    print("Aumentacion:", "desactivada" if args.sin_augment else AUGMENTACION_PLACAS)

    modelo.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        pretrained=False,
        project=str(MODELOS_DIR),
        name=args.name,
        exist_ok=True,
        patience=20,
        seed=42,
        deterministic=True,
        single_cls=False,
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=15,
        multi_scale=0.20,
        plots=True,
        val=True,
        **augmentacion,
    )

    metricas = modelo.val(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(MODELOS_DIR),
        name=f"{args.name}_val",
        exist_ok=True,
    )
    print(metricas)


if __name__ == "__main__":
    entrenar()
