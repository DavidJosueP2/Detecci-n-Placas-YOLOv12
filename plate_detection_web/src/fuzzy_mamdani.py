import json

import cv2
import numpy as np


INPUT_SETS = {
    "Muy segura": ("trap", [0, 0, 15, 20]),
    "Segura": ("tri", [15, 22, 30]),
    "Riesgo leve": ("tri", [25, 30, 35]),
    "Riesgo moderado": ("tri", [32, 40, 50]),
    "Riesgo grave": ("trap", [45, 55, 60, 60]),
}

OUTPUT_SETS = {
    "Normal": ("trap", [0, 0, 0.2, 0.8]),
    "Advertencia": ("tri", [0, 1, 2]),
    "Leve": ("tri", [2, 4, 6]),
    "Moderada": ("tri", [6, 9, 12]),
    "Grave": ("tri", [12, 18, 24]),
    "Muy grave": ("trap", [24, 36, 48, 48]),
}

RULES = [
    {"id": "R1", "if": "Muy segura", "then": "Normal"},
    {"id": "R2", "if": "Segura", "then": "Normal"},
    {"id": "R3", "if": "Riesgo leve", "then": "Advertencia"},
    {"id": "R4", "if": "Riesgo moderado", "then": "Moderada"},
    {"id": "R5", "if": "Riesgo grave", "then": "Grave"},
    {"id": "R6", "if": "Riesgo grave", "then": "Muy grave"},
]


def triangular(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b != a:
        y = np.maximum(y, np.minimum((x - a) / (b - a), 1.0))
    if c != b:
        y = np.minimum(y, np.maximum((c - x) / (c - b), 0.0))
    return np.clip(y, 0.0, 1.0)


def trapezoidal(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b == a:
        y = np.where(x <= b, 1.0, y)
    else:
        y = np.maximum(y, np.minimum((x - a) / (b - a), 1.0))
    y = np.where((x >= b) & (x <= c), 1.0, y)
    if d == c:
        y = np.where(x >= c, np.maximum(y, 1.0), y)
    else:
        y = np.minimum(y, np.maximum((d - x) / (d - c), 0.0))
    return np.clip(y, 0.0, 1.0)


def membership(kind, params, x):
    if kind == "tri":
        return triangular(x, *params)
    return trapezoidal(x, *params)


def evaluate_speed(velocidad_kmh, input_resolution=601, output_resolution=961):
    speed = float(max(0.0, min(60.0, velocidad_kmh)))
    x_input = np.linspace(0, 60, input_resolution)
    x_output = np.linspace(0, 48, output_resolution)

    input_memberships = {
        name: float(membership(kind, params, np.array([speed]))[0])
        for name, (kind, params) in INPUT_SETS.items()
    }

    output_curves = {
        name: membership(kind, params, x_output)
        for name, (kind, params) in OUTPUT_SETS.items()
    }

    rules = []
    aggregated = np.zeros_like(x_output)
    for rule in RULES:
        activation = input_memberships.get(rule["if"], 0.0)
        clipped = np.minimum(output_curves[rule["then"]], activation)
        aggregated = np.maximum(aggregated, clipped)
        rules.append(
            {
                "id": rule["id"],
                "antecedent": rule["if"],
                "consequent": rule["then"],
                "activation": round(float(activation), 4),
                "active": activation > 0,
            }
        )

    total_area = float(np.sum(aggregated))
    centroid = float(np.sum(x_output * aggregated) / total_area) if total_area > 0 else 0.0
    label = label_for_penalty(centroid)
    is_safe = label == "Normal" or centroid < 0.8
    penalty_hours = 0.0 if is_safe else centroid

    return {
        "velocidad_kmh": round(speed, 2),
        "input_universe": x_input.tolist(),
        "output_universe": x_output.tolist(),
        "input_sets": {
            name: membership(kind, params, x_input).tolist()
            for name, (kind, params) in INPUT_SETS.items()
        },
        "output_sets": {name: curve.tolist() for name, curve in output_curves.items()},
        "input_memberships": {
            name: round(value, 4) for name, value in input_memberships.items()
        },
        "rules": rules,
        "aggregated_output": aggregated.tolist(),
        "centroid": round(centroid, 3),
        "penalizacion_horas": round(penalty_hours, 2),
        "label": label,
        "is_safe": is_safe,
    }


def label_for_penalty(hours):
    values = {
        name: float(membership(kind, params, np.array([hours]))[0])
        for name, (kind, params) in OUTPUT_SETS.items()
    }
    return max(values, key=values.get)


def save_fuzzy_artifacts(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_graph": output_dir / "entrada_velocidad.png",
        "output_graph": output_dir / "salida_penalizacion.png",
        "aggregation_graph": output_dir / "agregacion_centroide.png",
        "json": output_dir / "mamdani.json",
    }

    _plot_input(result, paths["input_graph"])
    _plot_output(result, paths["output_graph"])
    _plot_aggregation(result, paths["aggregation_graph"])
    paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    return paths


def _plot_input(result, path):
    canvas = PlotCanvas("Entrada: velocidad km/h", "km/h", "pertenencia", 0, 60, 0, 1)
    x = np.array(result["input_universe"], dtype=float)
    for name, values in result["input_sets"].items():
        canvas.line(x, np.array(values, dtype=float), label=name)
    speed = result["velocidad_kmh"]
    canvas.vertical(speed, (24, 92, 255), f"{speed:.1f} km/h")
    for name, degree in result["input_memberships"].items():
        if degree > 0:
            canvas.text(f"{name}: {degree:.2f}")
    canvas.save(path)


def _plot_output(result, path):
    canvas = PlotCanvas("Salida: penalizacion administrativa", "horas", "pertenencia", 0, 48, 0, 1)
    x = np.array(result["output_universe"], dtype=float)
    for name, values in result["output_sets"].items():
        canvas.line(x, np.array(values, dtype=float), label=name)
    canvas.vertical(result["centroid"], (24, 92, 255), f"{result['centroid']:.2f} h")
    canvas.save(path)


def _plot_aggregation(result, path):
    canvas = PlotCanvas("Agregacion Mamdani y centroide", "horas", "pertenencia", 0, 48, 0, 1)
    x = np.array(result["output_universe"], dtype=float)
    y = np.array(result["aggregated_output"], dtype=float)
    canvas.fill(x, y, (8, 79, 153))
    canvas.line(x, y, color=(8, 79, 153), label="Salida agregada")
    canvas.vertical(result["centroid"], (24, 92, 255), f"{result['centroid']:.2f} h")
    canvas.text(f"Resultado: {result['label']}")
    canvas.text(f"Penalizacion: {result['penalizacion_horas']:.2f} h")
    canvas.save(path)


class PlotCanvas:
    COLORS = [
        (8, 79, 153),
        (11, 102, 195),
        (56, 217, 140),
        (245, 158, 11),
        (220, 38, 38),
        (124, 58, 237),
    ]

    def __init__(self, title, xlabel, ylabel, xmin, xmax, ymin, ymax, width=980, height=540):
        self.width = width
        self.height = height
        self.margin_left = 76
        self.margin_right = 28
        self.margin_top = 62
        self.margin_bottom = 62
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.color_index = 0
        self.note_y = self.margin_top + 28
        self.image = np.full((height, width, 3), 255, dtype=np.uint8)
        self._axes(title, xlabel, ylabel)

    def _axes(self, title, xlabel, ylabel):
        cv2.rectangle(
            self.image,
            (self.margin_left, self.margin_top),
            (self.width - self.margin_right, self.height - self.margin_bottom),
            (226, 232, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(self.image, title, (self.margin_left, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (15, 23, 42), 2, cv2.LINE_AA)
        cv2.putText(self.image, xlabel, (self.width // 2 - 35, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (71, 85, 105), 1, cv2.LINE_AA)
        cv2.putText(self.image, ylabel, (12, self.margin_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (71, 85, 105), 1, cv2.LINE_AA)
        for index in range(6):
            x = self.margin_left + int((self.width - self.margin_left - self.margin_right) * index / 5)
            value = self.xmin + (self.xmax - self.xmin) * index / 5
            cv2.line(self.image, (x, self.margin_top), (x, self.height - self.margin_bottom), (241, 245, 249), 1)
            cv2.putText(self.image, f"{value:.0f}", (x - 12, self.height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)
        for index in range(5):
            y = self.height - self.margin_bottom - int((self.height - self.margin_top - self.margin_bottom) * index / 4)
            value = self.ymin + (self.ymax - self.ymin) * index / 4
            cv2.line(self.image, (self.margin_left, y), (self.width - self.margin_right, y), (241, 245, 249), 1)
            cv2.putText(self.image, f"{value:.2f}", (28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA)

    def _point(self, x, y):
        px = self.margin_left + (x - self.xmin) / (self.xmax - self.xmin) * (self.width - self.margin_left - self.margin_right)
        py = self.height - self.margin_bottom - (y - self.ymin) / (self.ymax - self.ymin) * (self.height - self.margin_top - self.margin_bottom)
        return int(px), int(py)

    def line(self, x, y, color=None, label=None):
        color = color or self.COLORS[self.color_index % len(self.COLORS)]
        self.color_index += 1
        points = np.array([self._point(float(a), float(b)) for a, b in zip(x, y)], dtype=np.int32)
        cv2.polylines(self.image, [points], False, color, 2, cv2.LINE_AA)
        if label:
            lx = self.width - self.margin_right - 230
            ly = 70 + (self.color_index - 1) * 24
            cv2.line(self.image, (lx, ly - 5), (lx + 26, ly - 5), color, 2, cv2.LINE_AA)
            cv2.putText(self.image, label, (lx + 34, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (51, 65, 85), 1, cv2.LINE_AA)

    def fill(self, x, y, color):
        base = np.zeros_like(y)
        top = [self._point(float(a), float(b)) for a, b in zip(x, y)]
        bottom = [self._point(float(a), float(b)) for a, b in zip(x[::-1], base[::-1])]
        polygon = np.array(top + bottom, dtype=np.int32)
        overlay = self.image.copy()
        cv2.fillPoly(overlay, [polygon], color)
        self.image = cv2.addWeighted(overlay, 0.20, self.image, 0.80, 0)

    def vertical(self, x, color, label):
        p1 = self._point(x, self.ymin)
        p2 = self._point(x, self.ymax)
        cv2.line(self.image, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(self.image, label, (p2[0] + 8, max(22, p2[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    def text(self, value):
        cv2.putText(self.image, value, (self.margin_left + 12, self.note_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (51, 65, 85), 1, cv2.LINE_AA)
        self.note_y += 22

    def save(self, path):
        cv2.imwrite(str(path), self.image)
