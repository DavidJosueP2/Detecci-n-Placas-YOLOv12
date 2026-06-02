from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class PlateReader:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.25,
        image_size=320,
        device="auto",
        max_variants=4,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = None if device == "auto" else device
        self.max_variants = max(1, int(max_variants))

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo de caracteres configurado: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def read(self, crop):
        if crop is None or crop.size == 0:
            return "", 0.0, []

        chars = []
        for variant in self._prepare_variants(crop):
            for char in self._predict_chars(variant["image"]):
                char["weighted_confidence"] = char["confidence"] * variant["weight"]
                chars.append(char)

        chars = self._merge_character_candidates(chars)
        plate_chars = self._select_plate_line(chars)
        plate_chars = self._select_plate_sequence(plate_chars)
        text = "".join(char["value"] for char in plate_chars)

        if not text:
            text = self._normalize_plate_text("".join(char["value"] for char in chars))
            plate_chars = chars[: len(text)] if text else []

        confidence = (
            sum(char["confidence"] for char in plate_chars) / len(plate_chars)
            if plate_chars
            else 0.0
        )
        return text, confidence, plate_chars

    def preprocessing_debug(self, crop):
        if crop is None or crop.size == 0:
            return []

        steps = []

        def add(title, description, image, used=True):
            steps.append(
                {
                    "title": title,
                    "description": description,
                    "image": self._ensure_bgr(image),
                    "shape": f"{image.shape[1]} x {image.shape[0]} px",
                    "used": used,
                }
            )

        add("Recorte recibido", "Recorte actual generado por el detector de placa.", crop)
        crop = self._ensure_bgr(crop)
        add("Formato BGR", "Normaliza la imagen a 3 canales para que el modelo la reciba de forma consistente.", crop)

        crop = self._trim_plate_border(crop)
        add("Recorte de borde", "Retira un margen pequeno para reducir bordes de placa, fondo y ruido externo.", crop)

        height, width = crop.shape[:2]
        pad_x = max(4, int(width * 0.08))
        pad_y = max(4, int(height * 0.18))
        crop = cv2.copyMakeBorder(
            crop,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_REPLICATE,
        )
        add("Padding replicado", "Agrega margen alrededor de la placa para que los caracteres no queden pegados al borde.", crop)

        height, width = crop.shape[:2]
        scale = max(1.0, 190 / max(1, height), 520 / max(1, width))
        scale = min(scale, 4.0)
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        add("Escalado", f"Aumenta placas pequenas hasta un tamano util para OCR. Factor aplicado: {scale:.2f}x.", crop)

        crop = self._gray_world_balance(crop)
        add("Balance gray-world", "Corrige dominantes de color usando el promedio de los canales.", crop)

        crop = cv2.bilateralFilter(crop, 5, 55, 55)
        add("Filtro bilateral", "Reduce ruido conservando mejor los bordes de letras y numeros.", crop)

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l_channel)
        crop = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        add("CLAHE en luminancia", "Mejora contraste local sin depender tanto de la iluminacion global.", crop)

        blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
        base = cv2.addWeighted(crop, 1.55, blur, -0.55, 0)
        add("Acentuado de bordes", "Aplica enfoque por diferencia con desenfoque para resaltar detalles finos.", base)

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        adaptive = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )
        high_contrast = cv2.convertScaleAbs(base, alpha=1.18, beta=8)
        variants = [
            ("Variante OCR base", "Imagen final mejorada que se usa como primera entrada al OCR.", base),
            ("Variante alto contraste", "Refuerza contraste global con alpha/beta para letras suaves.", high_contrast),
            ("Variante gris CLAHE", "Convierte a escala de grises con contraste local reforzado.", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
            ("Variante binaria adaptativa", "Binariza segun iluminacion local y cierra pequenos cortes con morfologia.", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
        ]
        for index, (title, description, image) in enumerate(variants):
            add(title, description, image, used=index < self.max_variants)

        return steps

    def _predict_chars(self, prepared):
        result = self.model.predict(
            source=prepared,
            conf=max(0.08, self.confidence_threshold * 0.65),
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        chars = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            value = self._class_name(class_id)

            if confidence < self.confidence_threshold or not value:
                continue

            chars.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "value": value.upper(),
                    "width": max(1, x2 - x1),
                    "height": max(1, y2 - y1),
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                }
            )
        return chars

    @staticmethod
    def _normalize_plate_text(text):
        text = "".join(char for char in text.upper() if char.isalnum())
        if not text:
            return ""

        first_digit = next((index for index, char in enumerate(text) if char.isdigit()), None)
        if first_digit is None:
            return text[:7]

        letters = "".join(char for char in text[:first_digit] if char.isalpha())[:3]
        digits = "".join(char for char in text[first_digit:] if char.isdigit())[:4]

        if len(letters) >= 2 and len(digits) >= 2:
            return letters + digits

        return text[:7]

    def _prepare_variants(self, crop):
        base = self._prepare_base_crop(crop)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        adaptive = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )

        high_contrast = cv2.convertScaleAbs(base, alpha=1.18, beta=8)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

        variants = [
            {"image": base, "weight": 1.0},
            {"image": high_contrast, "weight": 0.95},
            {"image": gray_bgr, "weight": 0.9},
            {"image": adaptive_bgr, "weight": 0.45},
        ]
        return variants[: self.max_variants]

    def _prepare_base_crop(self, crop):
        crop = self._ensure_bgr(crop)
        crop = self._trim_plate_border(crop)
        height, width = crop.shape[:2]

        pad_x = max(4, int(width * 0.08))
        pad_y = max(4, int(height * 0.18))
        crop = cv2.copyMakeBorder(
            crop,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_REPLICATE,
        )

        height, width = crop.shape[:2]
        scale = max(1.0, 190 / max(1, height), 520 / max(1, width))
        scale = min(scale, 4.0)
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        crop = self._gray_world_balance(crop)
        crop = cv2.bilateralFilter(crop, 5, 55, 55)

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l_channel)
        crop = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
        crop = cv2.addWeighted(crop, 1.55, blur, -0.55, 0)
        return crop

    @staticmethod
    def _ensure_bgr(crop):
        if len(crop.shape) == 2:
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if crop.shape[2] == 4:
            return cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        return crop

    @staticmethod
    def _trim_plate_border(crop):
        height, width = crop.shape[:2]
        if height < 24 or width < 72:
            return crop

        trim_x = int(width * 0.02)
        trim_y = int(height * 0.03)
        if trim_x == 0 and trim_y == 0:
            return crop

        return crop[
            trim_y : height - trim_y if trim_y else height,
            trim_x : width - trim_x if trim_x else width,
        ]

    @staticmethod
    def _gray_world_balance(image):
        image_float = image.astype(np.float32)
        means = image_float.reshape(-1, 3).mean(axis=0)
        gray = means.mean()
        gains = gray / np.maximum(means, 1.0)
        balanced = np.clip(image_float * gains, 0, 255)
        return balanced.astype(np.uint8)

    def _merge_character_candidates(self, chars):
        groups = []
        for char in sorted(chars, key=lambda item: item["center_x"]):
            for group in groups:
                if self._same_character_position(char, group):
                    group.append(char)
                    break
            else:
                groups.append([char])

        return sorted(
            [self._collapse_character_group(group) for group in groups],
            key=lambda item: item["x1"],
        )

    def _same_character_position(self, char, group):
        for item in group:
            if self._iou(char, item) > 0.18:
                return True
            center_close = abs(char["center_x"] - item["center_x"]) <= max(
                8,
                min(char["width"], item["width"]) * 0.45,
            )
            same_line = abs(char["center_y"] - item["center_y"]) <= max(
                12,
                max(char["height"], item["height"]) * 0.6,
            )
            if center_close and same_line:
                return True
        return False

    def _collapse_character_group(self, group):
        alternatives = {}
        for char in group:
            alternative = alternatives.setdefault(
                char["value"],
                {"value": char["value"], "score": 0.0, "confidence": 0.0},
            )
            alternative["score"] += char.get("weighted_confidence", char["confidence"])
            alternative["confidence"] = max(alternative["confidence"], char["confidence"])

        ordered_alternatives = sorted(
            alternatives.values(),
            key=lambda item: (item["score"], item["confidence"]),
            reverse=True,
        )
        best = ordered_alternatives[0]
        best_items = [item for item in group if item["value"] == best["value"]]

        return {
            "x1": min(item["x1"] for item in best_items),
            "y1": min(item["y1"] for item in best_items),
            "x2": max(item["x2"] for item in best_items),
            "y2": max(item["y2"] for item in best_items),
            "confidence": best["confidence"],
            "value": best["value"],
            "width": max(item["width"] for item in best_items),
            "height": max(item["height"] for item in best_items),
            "center_x": sum(item["center_x"] for item in best_items) / len(best_items),
            "center_y": sum(item["center_y"] for item in best_items) / len(best_items),
            "alternatives": ordered_alternatives,
        }

    @staticmethod
    def _iou(first, second):
        x1 = max(first["x1"], second["x1"])
        y1 = max(first["y1"], second["y1"])
        x2 = min(first["x2"], second["x2"])
        y2 = min(first["y2"], second["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        first_area = max(1, first["x2"] - first["x1"]) * max(1, first["y2"] - first["y1"])
        second_area = max(1, second["x2"] - second["x1"]) * max(1, second["y2"] - second["y1"])
        return inter / (first_area + second_area - inter)

    def _select_plate_line(self, chars):
        if not chars:
            return []

        heights = sorted(char["height"] for char in chars)
        median_height = heights[len(heights) // 2]
        reference_height = heights[int(len(heights) * 0.75)]
        large_chars = [
            char
            for char in chars
            if char["height"] >= max(8, reference_height * 0.62)
        ]
        if not large_chars:
            large_chars = chars

        lines = []
        for char in sorted(large_chars, key=lambda item: item["center_y"]):
            for line in lines:
                line_center = sum(item["center_y"] for item in line) / len(line)
                if abs(char["center_y"] - line_center) <= max(10, median_height * 0.75):
                    line.append(char)
                    break
            else:
                lines.append([char])

        best_line = max(
            lines,
            key=lambda line: (
                len(line),
                sum(item["height"] for item in line) / len(line),
                sum(item["center_y"] for item in line) / len(line),
            ),
        )
        return sorted(best_line, key=lambda item: item["x1"])

    def _select_plate_sequence(self, chars):
        chars = sorted(chars, key=lambda item: item["x1"])
        if not chars:
            return []

        best_candidate = None
        for letter_count, digit_count in ((3, 4), (3, 3)):
            length = letter_count + digit_count
            if len(chars) < length:
                continue

            for indices in combinations(range(len(chars)), length):
                selected = []
                score = 0.0

                for offset, index in enumerate(indices):
                    char = chars[index]
                    expected = "letter" if offset < letter_count else "digit"
                    alternative = self._best_alternative(char, expected)
                    if alternative is None:
                        break

                    selected_char = dict(char)
                    selected_char["value"] = alternative["value"]
                    selected_char["confidence"] = alternative["confidence"]
                    selected.append(selected_char)
                    score += alternative["score"]
                else:
                    avg_confidence = sum(item["confidence"] for item in selected) / length
                    median_selected_height = sorted(
                        item["height"] for item in selected
                    )[length // 2]
                    height_penalty = sum(
                        max(0.0, (median_selected_height * 0.68 - item["height"]) / median_selected_height)
                        for item in selected
                    )
                    skipped_inside = (indices[-1] - indices[0] + 1) - length
                    score = (score / length) + avg_confidence * 0.75
                    score -= skipped_inside * 0.12
                    score -= height_penalty * 0.35
                    if digit_count == 4:
                        score += 0.08
                    candidate = (score, selected)
                    if best_candidate is None or candidate[0] > best_candidate[0]:
                        best_candidate = candidate

        if best_candidate is not None:
            return best_candidate[1]

        text = self._normalize_plate_text("".join(char["value"] for char in chars))
        return chars[: len(text)] if text else []

    @staticmethod
    def _best_alternative(char, expected):
        alternatives = char.get("alternatives") or [
            {
                "value": char["value"],
                "score": char.get("weighted_confidence", char["confidence"]),
                "confidence": char["confidence"],
            }
        ]

        if expected == "letter":
            filtered = [item for item in alternatives if item["value"].isalpha()]
        else:
            filtered = [item for item in alternatives if item["value"].isdigit()]

        if not filtered:
            return None

        return max(filtered, key=lambda item: (item["score"], item["confidence"]))

    def _class_name(self, class_id):
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, ""))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return ""
