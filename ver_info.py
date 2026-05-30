import torch
from pathlib import Path
from pprint import pformat
import json

# Cambia aquí el modelo que quieres revisar
ruta = Path("models/Best_epoch_1_90mil.pt")
# ruta = Path("modelos/placas_yolo12_pro/weights/last.pt")
# ruta = Path("models/lp_yolo11x_morsetech.pt")

salida_txt = Path("reporte/reporte_completo_pt.txt")
salida_json = Path("reporte/reporte_resumen_pt.json")

ckpt = torch.load(ruta, map_location="cpu", weights_only=False)

lineas = []
resumen = {}

def escribir(texto=""):
    lineas.append(str(texto))

def tipo_valor(v):
    try:
        return str(type(v))
    except Exception:
        return "tipo_desconocido"

def valor_simple(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return None

def imprimir_objeto(nombre, obj, nivel=0, max_nivel=8, visitados=None):
    if visitados is None:
        visitados = set()

    indent = "  " * nivel

    try:
        obj_id = id(obj)
        if obj_id in visitados:
            escribir(f"{indent}{nombre}: <referencia repetida> tipo={type(obj)}")
            return
        visitados.add(obj_id)
    except Exception:
        pass

    escribir(f"{indent}{nombre}: tipo={type(obj)}")

    if nivel > max_nivel:
        escribir(f"{indent}  <máxima profundidad alcanzada>")
        return

    if isinstance(obj, dict):
        escribir(f"{indent}  tamaño_dict={len(obj)}")
        for k, v in obj.items():
            escribir(f"{indent}  clave: {repr(k)} | tipo={type(v)}")

            if isinstance(v, (str, int, float, bool)) or v is None:
                escribir(f"{indent}    valor: {repr(v)}")
            elif isinstance(v, (dict, list, tuple)):
                imprimir_objeto(f"{nombre}.{k}", v, nivel + 2, max_nivel, visitados)
            else:
                imprimir_atributos(f"{nombre}.{k}", v, nivel + 2, max_nivel, visitados)

    elif isinstance(obj, (list, tuple)):
        escribir(f"{indent}  tamaño_lista={len(obj)}")
        for i, v in enumerate(obj[:100]):
            if isinstance(v, (str, int, float, bool)) or v is None:
                escribir(f"{indent}  [{i}] {repr(v)}")
            else:
                imprimir_objeto(f"{nombre}[{i}]", v, nivel + 1, max_nivel, visitados)

        if len(obj) > 100:
            escribir(f"{indent}  ... se omitieron {len(obj) - 100} elementos")

    else:
        imprimir_atributos(nombre, obj, nivel, max_nivel, visitados)

def imprimir_atributos(nombre, obj, nivel=0, max_nivel=8, visitados=None):
    indent = "  " * nivel

    escribir(f"{indent}{nombre}: tipo={type(obj)}")

    # Intentar imprimir representación corta
    try:
        rep = repr(obj)
        if len(rep) > 1000:
            rep = rep[:1000] + " ... <recortado>"
        escribir(f"{indent}  repr: {rep}")
    except Exception as e:
        escribir(f"{indent}  repr_error: {e}")

    # Atributos importantes conocidos en modelos YOLO
    atributos_interes = [
        "names", "yaml", "args", "stride", "task", "model", "nc",
        "training", "save", "device", "hyp", "criterion"
    ]

    for attr in atributos_interes:
        try:
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                escribir(f"{indent}  atributo {attr}: tipo={type(v)}")
                if isinstance(v, (str, int, float, bool)) or v is None:
                    escribir(f"{indent}    valor: {repr(v)}")
                else:
                    texto = pformat(v, width=120)
                    if len(texto) > 5000:
                        texto = texto[:5000] + "\n... <recortado>"
                    escribir(f"{indent}    {texto}")
        except Exception as e:
            escribir(f"{indent}  atributo {attr}: error={e}")

    # Todos los atributos públicos posibles
    try:
        attrs = [a for a in dir(obj) if not a.startswith("_")]
        escribir(f"{indent}  atributos_publicos_total={len(attrs)}")
        escribir(f"{indent}  atributos_publicos:")
        escribir(f"{indent}    {attrs}")

        for attr in attrs:
            try:
                v = getattr(obj, attr)

                if callable(v):
                    continue

                escribir(f"{indent}  .{attr}: tipo={type(v)}")

                if isinstance(v, (str, int, float, bool)) or v is None:
                    escribir(f"{indent}    valor: {repr(v)}")
                elif isinstance(v, dict):
                    texto = pformat(v, width=120)
                    if len(texto) > 3000:
                        texto = texto[:3000] + "\n... <recortado>"
                    escribir(f"{indent}    {texto}")
                elif isinstance(v, (list, tuple)):
                    escribir(f"{indent}    tamaño={len(v)}")
                    texto = pformat(v[:20] if len(v) > 20 else v, width=120)
                    if len(texto) > 3000:
                        texto = texto[:3000] + "\n... <recortado>"
                    escribir(f"{indent}    {texto}")
                else:
                    rep = repr(v)
                    if len(rep) > 1000:
                        rep = rep[:1000] + " ... <recortado>"
                    escribir(f"{indent}    repr: {rep}")

            except Exception as e:
                escribir(f"{indent}  .{attr}: error={e}")

    except Exception as e:
        escribir(f"{indent}  dir_error: {e}")

def convertir_json_seguro(obj, nivel=0, max_nivel=5, visitados=None):
    if visitados is None:
        visitados = set()

    if nivel > max_nivel:
        return "<maxima_profundidad>"

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    try:
        obj_id = id(obj)
        if obj_id in visitados:
            return "<referencia_repetida>"
        visitados.add(obj_id)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {
            str(k): convertir_json_seguro(v, nivel + 1, max_nivel, visitados)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            convertir_json_seguro(v, nivel + 1, max_nivel, visitados)
            for v in obj[:50]
        ]

    datos = {
        "__tipo__": str(type(obj))
    }

    for attr in ["names", "yaml", "args", "stride", "task", "nc"]:
        try:
            if hasattr(obj, attr):
                datos[attr] = convertir_json_seguro(getattr(obj, attr), nivel + 1, max_nivel, visitados)
        except Exception as e:
            datos[attr] = f"<error: {e}>"

    return datos

# ==========================
# REPORTE
# ==========================

escribir("============================================================")
escribir("REPORTE COMPLETO DEL CHECKPOINT .PT")
escribir("============================================================")
escribir(f"Archivo: {ruta}")
escribir(f"Existe: {ruta.exists()}")

if ruta.exists():
    escribir(f"Tamaño bytes: {ruta.stat().st_size}")
    escribir(f"Tamaño MB: {ruta.stat().st_size / (1024 * 1024):.2f}")

escribir("")
escribir("============================================================")
escribir("TIPO RAÍZ")
escribir("============================================================")
escribir(type(ckpt))

escribir("")
escribir("============================================================")
escribir("INSPECCIÓN RECURSIVA")
escribir("============================================================")
imprimir_objeto("ckpt", ckpt, max_nivel=8)

escribir("")
escribir("============================================================")
escribir("BÚSQUEDA DE PALABRAS RELACIONADAS CON DATASET")
escribir("============================================================")

palabras = [
    "data", "dataset", "image", "images", "img", "label", "labels",
    "train", "val", "test", "nc", "names", "class", "classes",
    "sample", "samples", "instance", "instances", "batch", "epoch",
    "fitness", "metrics", "results"
]

def buscar_palabras(obj, ruta_actual="ckpt", nivel=0, max_nivel=10, visitados=None):
    if visitados is None:
        visitados = set()

    if nivel > max_nivel:
        return

    try:
        obj_id = id(obj)
        if obj_id in visitados:
            return
        visitados.add(obj_id)
    except Exception:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k).lower()
            if any(p in k_str for p in palabras):
                escribir(f"\n{ruta_actual}.{k}")
                escribir(f"Tipo: {type(v)}")
                try:
                    texto = pformat(v, width=120)
                    if len(texto) > 8000:
                        texto = texto[:8000] + "\n... <recortado>"
                    escribir(texto)
                except Exception as e:
                    escribir(f"No se pudo imprimir: {e}")

            buscar_palabras(v, f"{ruta_actual}.{k}", nivel + 1, max_nivel, visitados)

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:100]):
            buscar_palabras(v, f"{ruta_actual}[{i}]", nivel + 1, max_nivel, visitados)

    else:
        try:
            attrs = [a for a in dir(obj) if not a.startswith("_")]
            for attr in attrs:
                attr_low = attr.lower()
                if any(p in attr_low for p in palabras):
                    try:
                        v = getattr(obj, attr)
                        if callable(v):
                            continue
                        escribir(f"\n{ruta_actual}.{attr}")
                        escribir(f"Tipo: {type(v)}")
                        texto = pformat(v, width=120)
                        if len(texto) > 8000:
                            texto = texto[:8000] + "\n... <recortado>"
                        escribir(texto)
                    except Exception as e:
                        escribir(f"{ruta_actual}.{attr}: error={e}")
        except Exception:
            pass

buscar_palabras(ckpt)

# Guardar TXT
salida_txt.write_text("\n".join(lineas), encoding="utf-8")

# Guardar JSON resumen
resumen = convertir_json_seguro(ckpt, max_nivel=6)
with open(salida_json, "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print(f"Listo. Reporte completo guardado en: {salida_txt}")
print(f"Resumen JSON guardado en: {salida_json}")