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

Tambien puedes configurar todo en `.env` dentro de esta carpeta. La app lo carga automaticamente al iniciar:

```text
VIDEO_SOURCE=0
PLATE_MODEL_PATH=models/90mil_38 epocas.pt
DETECTION_EVERY_N_FRAMES=6
OCR_VARIANTS=4
```

Otros parametros configurables:

```powershell
$env:CONFIDENCE_THRESHOLD="0.25"
$env:IMAGE_SIZE="640"
$env:STREAM_TARGET_FPS="12"
$env:STREAM_MAX_WIDTH="960"
$env:DETECTION_EVERY_N_FRAMES="6"
$env:OCR_INTERVAL_SECONDS="1.25"
$env:OCR_RETRY_INTERVAL_SECONDS="0.30"
$env:OCR_VARIANTS="4"
$env:SPEED_LINE_A_LEFT_Y="0.36"
$env:SPEED_LINE_A_RIGHT_Y="0.32"
$env:SPEED_LINE_B_LEFT_Y="0.66"
$env:SPEED_LINE_B_RIGHT_Y="0.63"
$env:SPEED_ROI_X1="0.0"
$env:SPEED_ROI_X2="1.0"
$env:SPEED_DISTANCE_METERS="5.0"
$env:SPEED_DIRECTION="both"
$env:YOLO_DEVICE="auto"
$env:YOLO_TRACKER="bytetrack.yaml"
$env:IOU_THRESHOLD="0.55"
```

`YOLO_DEVICE` puede ser `auto`, `cpu`, `0` o `cuda:0`, segun tu entorno.

Para mantener videos locales mas fluidos sin saltar cuadros, el stream muestra todos los frames y ejecuta YOLO cada `DETECTION_EVERY_N_FRAMES`, reutilizando la ultima caja detectada entre inferencias. El OCR intenta leer rapido mientras una placa no tiene texto (`OCR_RETRY_INTERVAL_SECONDS`) y despues usa cache con refrescos periodicos (`OCR_INTERVAL_SECONDS`) para no frenar cada frame.

La velocidad se calcula solo cuando una misma placa cruza dos lineas configurables. Cada linea tiene una altura normalizada en el lado izquierdo y otra en el lado derecho: `SPEED_LINE_A_LEFT_Y`, `SPEED_LINE_A_RIGHT_Y`, `SPEED_LINE_B_LEFT_Y` y `SPEED_LINE_B_RIGHT_Y`. `0.0` es arriba del video y `1.0` es abajo; si `RIGHT_Y` es menor que `LEFT_Y`, la linea sube hacia la derecha. `SPEED_ROI_X1` y `SPEED_ROI_X2` definen el rango horizontal de medicion y dibujo. `SPEED_DISTANCE_METERS` debe ser la distancia real medida entre esas dos lineas sobre la via.

Cuando una placa ya tiene velocidad, el sistema evalua esa velocidad con un Mamdani tipo 1. Si el resultado es `Normal`, no se guarda incidencia ni se muestra penalizacion. Si devuelve `Advertencia`, `Leve`, `Moderada`, `Grave` o `Muy grave`, se guarda una incidencia en SQLite con frame, recorte de placa, datos simulados del vehiculo, JSON explicativo y graficas PNG en `static/incidencias/<id>/`. El historial queda disponible en `/incidencias`.

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
