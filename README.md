# Deteccion de Placas con YOLO

Proyecto para entrenar y probar modelos YOLO de deteccion de placas.

## Instalacion

Crear entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Si tienes GPU NVIDIA, instala PyTorch con CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

Luego instala dependencias del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si vas a usar solo CPU, puedes omitir `requirements-cuda.txt` e instalar PyTorch CPU siguiendo la guia oficial de PyTorch.

## Probar una imagen

```powershell
.\.venv\Scripts\python.exe probar_imagen.py
```

O con una ruta directa:

```powershell
.\.venv\Scripts\python.exe probar_imagen.py "ruta\imagen.jpg"
```

## Entrenar modelo pro

El dataset no va incluido en Git. Coloca el dataset en:

```text
datasets/License Plate Recognition/
```

El YAML recomendado para entrenar es:

```text
license_plate_recognition.yaml
```

Antes de entrenar, verifica que PyTorch detecta la GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### PC potente, modo automatico recomendado

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --device auto --batch -1 --workers 8 --imgsz 768 --epochs 50 --patience 10
```

Esto usa:

```text
modelo base: yolo12s.pt
GPU: automatico si hay CUDA
batch: automatico
workers: 8
tamano imagen: 768
epocas: 50
early stopping: 10 epocas sin mejora
```

### Prueba rapida antes del entrenamiento largo

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --device auto --quick --fraction 0.01 --batch 8 --workers 4 --name placas_yolo12_pro_smoke
```

### PC con GPU limitada, comando usado localmente

Para una GPU de 4 GB, usar configuracion conservadora:

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --device 0 --batch 4 --workers 0 --imgsz 640 --epochs 50 --patience 10
```

Si da error de memoria CUDA, bajar a:

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --device 0 --batch 2 --workers 0 --imgsz 640 --epochs 50 --patience 10
```

### Mas precision si la GPU aguanta

Usar modelo mediano:

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --model yolo12m.pt --device auto --batch -1 --workers 8 --imgsz 768 --epochs 50 --patience 10 --name placas_yolo12m_pro
```

Usar imagen mas grande para placas pequenas:

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --model yolo12s.pt --device auto --batch -1 --workers 8 --imgsz 960 --epochs 50 --patience 10 --name placas_yolo12s_960
```

### Si hay error de memoria CUDA

Bajar batch:

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --device 0 --batch 8 --workers 4
```

Si sigue fallando:

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --device 0 --batch 4 --workers 2
```

### Reanudar entrenamiento interrumpido

```powershell
.\.venv\Scripts\python.exe entrenamiento_pro.py --data license_plate_recognition.yaml --resume
```

## Modelos

Los checkpoints `.pt` si pueden versionarse si pesan menos de 100 MB. Si alguno supera ese limite, usar Git LFS o subirlo como release asset.

El modelo entrenado final queda normalmente en:

```text
modelos/placas_yolo12_pro/weights/best.pt
```

Si el entrenamiento se hace en otra PC, basta con copiar ese `best.pt` a esta maquina y usarlo para inferencia. Por ejemplo, puedes guardarlo como:

```text
modelos/placas_yolo12_pro/weights/best.pt
```

o pasarlo explicitamente al script:

```powershell
.\.venv\Scripts\python.exe probar_imagen.py "ruta\imagen.jpg" --model "ruta\best.pt"
```
