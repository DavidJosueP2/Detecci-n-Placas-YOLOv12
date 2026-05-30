# Plate Detection Web

Primera fase del sistema web para deteccion y seguimiento de placas vehiculares en tiempo real con Flask, OpenCV y Ultralytics YOLO.

## Instalacion

Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Modelo

Coloca tu modelo entrenado en:

```text
models/plate_model.pt
```

Si ese archivo no existe, la app intenta usar `models/lp_yolo11x_morsetech.pt` y luego los modelos disponibles en la carpeta raiz del proyecto.

Tambien puedes usar una ruta externa:

```powershell
$env:PLATE_MODEL_PATH="D:\ruta\modelo.pt"
```

## Fuente de video

Por defecto usa la camara `0`. Para cambiarla:

```powershell
$env:VIDEO_SOURCE="0"
```

O para usar un video local:

```powershell
$env:VIDEO_SOURCE="D:\ruta\video.mp4"
```

Otros parametros configurables:

```powershell
$env:CONFIDENCE_THRESHOLD="0.25"
$env:IMAGE_SIZE="640"
$env:YOLO_DEVICE="auto"
$env:YOLO_TRACKER="bytetrack.yaml"
$env:IOU_THRESHOLD="0.55"
```

`YOLO_DEVICE` puede ser `auto`, `cpu`, `0` o `cuda:0`, segun tu entorno.

El seguimiento usa `model.track(..., persist=True)` para mantener el ID de la placa entre frames. Puedes cambiar el tracker a `botsort.yaml` si el video tiene oclusiones o movimientos mas complejos.

## Cargar videos

La interfaz permite arrastrar y soltar videos directamente. Los archivos se guardan en:

```text
static/captures/
```

La fuente del stream se actualiza automaticamente al video cargado. Tambien puedes volver a la camara desde el boton de la interfaz.

## Ejecutar

```powershell
.\.venv\Scripts\python.exe app.py
```

Luego abre:

```text
http://127.0.0.1:5000
```

## Pendiente para siguientes fases

- OCR de placa.
- Estimacion de velocidad.
- Sistema difuso Mamdani tipo 1.
- Registro de infracciones.

Los puntos de integracion estan separados en `src/frame_processor.py` y `src/video_stream.py` para conectar esos modulos despues.
