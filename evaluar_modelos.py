"""
Evaluación comparativa de todos los modelos de detección de placas.

Evalúa cada modelo contra los datasets de validación (License Plate Recognition
y Placas Ecuador) y genera un reporte con métricas: Precision, Recall, mAP50,
mAP50-95, y velocidad de inferencia.

Uso:
    python evaluar_modelos.py                        # Evalúa todos los modelos en ambos datasets
    python evaluar_modelos.py --dataset lpr           # Solo License Plate Recognition
    python evaluar_modelos.py --dataset ecuador        # Solo Placas Ecuador
    python evaluar_modelos.py --modelos best_model.pt  # Un modelo específico
    python evaluar_modelos.py --verbose               # Mostrar salida detallada de YOLO
"""

from pathlib import Path
import argparse
import json
import sys
import time
from datetime import datetime

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs_yolo11_license_plate"
RESULTADOS_DIR = ROOT / "resultados_evaluacion"

# ──────────────────────────────────────────────────────────────────────
# Datasets disponibles
# ──────────────────────────────────────────────────────────────────────
DATASETS = {
    "License Plate Recognition": {
        "yaml": ROOT / "license_plate_recognition.yaml",
        "short": "lpr",
    },
    "Placas Ecuador": {
        "yaml": ROOT / "placas_ecuador_preparado.yaml",
        "short": "ecuador",
    },
}

# ──────────────────────────────────────────────────────────────────────
# Descubrimiento automático de modelos .pt
# ──────────────────────────────────────────────────────────────────────
def descubrir_modelos() -> dict[str, Path]:
    """Busca todos los best.pt y .pt sueltos en las carpetas de modelos."""
    modelos = {}

    # 1. modelos/<experimento>/weights/best.pt  (entrenados con entrenamiento_pro/yolo12)
    if MODELOS_DIR.exists():
        for exp in sorted(MODELOS_DIR.iterdir()):
            best = exp / "weights" / "best.pt"
            if best.exists():
                modelos[f"modelos/{exp.name}/best.pt"] = best

    # 2. outputs_yolo11_license_plate/<exp>/weights/best.pt
    if OUTPUTS_DIR.exists():
        for exp in sorted(OUTPUTS_DIR.iterdir()):
            best = exp / "weights" / "best.pt"
            if best.exists():
                modelos[f"outputs_yolo11/{exp.name}/best.pt"] = best

    # 3. models/*.pt  (modelos sueltos descargados o copiados)
    if MODELS_DIR.exists():
        for pt in sorted(MODELS_DIR.glob("*.pt")):
            modelos[f"models/{pt.name}"] = pt

    # 4. yolo12s.pt en la raíz (base preentrenado)
    base_pt = ROOT / "yolo12s.pt"
    if base_pt.exists():
        modelos["yolo12s.pt (base)"] = base_pt

    return modelos


def evaluar_modelo(modelo_path: Path, data_yaml: Path, device, imgsz=640, batch=4, verbose=False):
    """Ejecuta model.val() y devuelve un dict con las métricas."""
    try:
        modelo = YOLO(str(modelo_path))
    except Exception as e:
        return {"error": f"No se pudo cargar: {e}"}

    t0 = time.time()
    try:
        metricas = modelo.val(
            data=str(data_yaml),
            imgsz=imgsz,
            batch=batch,
            workers=0,
            device=device,
            verbose=verbose,
            plots=True,
            project=str(RESULTADOS_DIR),
            name=f"{modelo_path.stem}_{data_yaml.stem}",
            exist_ok=True,
        )
    except Exception as e:
        return {"error": f"Error en validación: {e}"}
    dt = time.time() - t0

    # Extraer métricas del objeto Results
    try:
        box = metricas.box
        result = {
            "precision": round(float(box.mp), 5),
            "recall": round(float(box.mr), 5),
            "mAP50": round(float(box.map50), 5),
            "mAP50-95": round(float(box.map), 5),
            "tiempo_val_seg": round(dt, 2),
        }
        # Velocidad de inferencia (ms por imagen)
        if hasattr(metricas, "speed") and metricas.speed:
            speed = metricas.speed
            result["preprocess_ms"] = round(speed.get("preprocess", 0), 2)
            result["inference_ms"] = round(speed.get("inference", 0), 2)
            result["postprocess_ms"] = round(speed.get("postprocess", 0), 2)
        return result
    except Exception as e:
        return {"error": f"Error extrayendo métricas: {e}"}


def imprimir_tabla(resultados: list[dict]):
    """Imprime una tabla bonita en consola."""
    if not resultados:
        print("  No hay resultados para mostrar.")
        return

    # Encabezados
    cols = [
        ("Modelo", 45),
        ("Dataset", 28),
        ("Precision", 10),
        ("Recall", 10),
        ("mAP50", 10),
        ("mAP50-95", 10),
        ("Infer(ms)", 10),
        ("Tiempo(s)", 10),
    ]
    header = " | ".join(f"{name:<{w}}" for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)

    print()
    print(header)
    print(sep)

    for r in resultados:
        if "error" in r:
            vals = [
                f"{r['modelo']:<45}",
                f"{r['dataset']:<28}",
                f"{'ERROR':<10}",
                f"{'':<10}",
                f"{'':<10}",
                f"{'':<10}",
                f"{'':<10}",
                f"{'':<10}",
            ]
            print(" | ".join(vals))
            print(f"  → {r['error']}")
        else:
            vals = [
                f"{r['modelo']:<45}",
                f"{r['dataset']:<28}",
                f"{r.get('precision', 'N/A'):>10}",
                f"{r.get('recall', 'N/A'):>10}",
                f"{r.get('mAP50', 'N/A'):>10}",
                f"{r.get('mAP50-95', 'N/A'):>10}",
                f"{r.get('inference_ms', 'N/A'):>10}",
                f"{r.get('tiempo_val_seg', 'N/A'):>10}",
            ]
            print(" | ".join(vals))

    print()


def guardar_reporte(resultados: list[dict], ruta: Path):
    """Guarda los resultados en un JSON legible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en: {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelos de placas contra datasets de validación.")
    parser.add_argument("--dataset", choices=["lpr", "ecuador", "todos"], default="todos",
                        help="Dataset a evaluar: lpr, ecuador o todos.")
    parser.add_argument("--modelos", nargs="*", default=None,
                        help="Nombres específicos de .pt a evaluar (ej: best_model.pt). Si no se pasa, evalúa todos.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu o 0.")
    parser.add_argument("--verbose", action="store_true", help="Mostrar salida detallada.")
    args = parser.parse_args()

    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print("=" * 120)
    print(f"  EVALUACIÓN COMPARATIVA DE MODELOS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dispositivo: {'CUDA (GPU)' if device == 0 else device}")
    print("=" * 120)

    # Descubrir modelos
    todos_modelos = descubrir_modelos()
    if args.modelos:
        # Filtrar por nombres parciales
        filtrado = {}
        for nombre, path in todos_modelos.items():
            for filtro in args.modelos:
                if filtro.lower() in nombre.lower() or filtro.lower() in path.name.lower():
                    filtrado[nombre] = path
                    break
        todos_modelos = filtrado

    if not todos_modelos:
        print("\n  No se encontraron modelos .pt para evaluar.")
        print("  Carpetas buscadas: modelos/, models/, outputs_yolo11_license_plate/")
        return 1

    print(f"\n  Modelos encontrados: {len(todos_modelos)}")
    for nombre, ruta in todos_modelos.items():
        size_mb = ruta.stat().st_size / (1024 * 1024)
        print(f"    • {nombre}  ({size_mb:.1f} MB)")

    # Seleccionar datasets
    if args.dataset == "todos":
        datasets_eval = DATASETS
    else:
        datasets_eval = {k: v for k, v in DATASETS.items() if v["short"] == args.dataset}

    print(f"\n  Datasets de validación: {len(datasets_eval)}")
    for nombre, info in datasets_eval.items():
        yaml_existe = "✓" if info["yaml"].exists() else "✗ NO ENCONTRADO"
        print(f"    • {nombre}  [{yaml_existe}]")

    # Evaluar
    resultados = []
    total = len(todos_modelos) * len(datasets_eval)
    actual = 0

    for modelo_nombre, modelo_path in todos_modelos.items():
        for ds_nombre, ds_info in datasets_eval.items():
            actual += 1
            yaml_path = ds_info["yaml"]

            print(f"\n{'─' * 120}")
            print(f"  [{actual}/{total}] Evaluando: {modelo_nombre}")
            print(f"       Dataset: {ds_nombre}")
            print(f"       YAML: {yaml_path}")

            if not yaml_path.exists():
                r = {
                    "modelo": modelo_nombre,
                    "modelo_path": str(modelo_path),
                    "dataset": ds_nombre,
                    "error": f"YAML no encontrado: {yaml_path}",
                }
                resultados.append(r)
                print(f"  ⚠️  YAML no encontrado, saltando...")
                continue

            metricas = evaluar_modelo(
                modelo_path, yaml_path, device,
                imgsz=args.imgsz, batch=args.batch, verbose=args.verbose,
            )

            r = {
                "modelo": modelo_nombre,
                "modelo_path": str(modelo_path),
                "dataset": ds_nombre,
                **metricas,
            }
            resultados.append(r)

            if "error" in metricas:
                print(f"  ❌ Error: {metricas['error']}")
            else:
                print(f"  ✅ Precision={metricas['precision']:.4f}  Recall={metricas['recall']:.4f}  "
                      f"mAP50={metricas['mAP50']:.4f}  mAP50-95={metricas['mAP50-95']:.4f}  "
                      f"Inferencia={metricas.get('inference_ms', '?')}ms")

    # Resumen
    print(f"\n{'=' * 120}")
    print("  RESUMEN DE RESULTADOS")
    print(f"{'=' * 120}")
    imprimir_tabla(resultados)

    # Encontrar el mejor modelo por dataset
    for ds_nombre in datasets_eval:
        ds_results = [r for r in resultados if r["dataset"] == ds_nombre and "error" not in r]
        if ds_results:
            mejor = max(ds_results, key=lambda x: x.get("mAP50-95", 0))
            print(f"  🏆 Mejor modelo en '{ds_nombre}': {mejor['modelo']}")
            print(f"     mAP50-95={mejor['mAP50-95']:.4f}  mAP50={mejor['mAP50']:.4f}")
            print()

    # Guardar reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporte_path = RESULTADOS_DIR / f"evaluacion_{timestamp}.json"
    guardar_reporte(resultados, reporte_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
